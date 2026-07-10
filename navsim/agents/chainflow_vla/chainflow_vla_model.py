from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from navsim.agents.chainflow_vla.utils import pylogger

from .decoders.decoder_score.score import ScoreDecoder
from .decoders.decoder_traj import TrajectoryDecoder
from .encoders.image.dinov2_lora import ImgEncoder


log = pylogger.get_pylogger(__name__)


CAMERA_CONFIG_KEYS: Tuple[str, ...] = (
    "cam_f0",
    "cam_l0",
    "cam_l1",
    "cam_l2",
    "cam_r0",
    "cam_r1",
    "cam_r2",
    "cam_b0",
)


class ChainFlowVLAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model
        self.b2d = bool(getattr(config, "b2d", False))

        self.num_cams = self._count_enabled_sensors(CAMERA_CONFIG_KEYS)
        self.num_lidar = 1 if len(self._config["lidar_pc"]) > 0 else 0

        self._build_scene_backbones(config)
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11 * 4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        self.trajectory_decoder = TrajectoryDecoder(config)
        self._use_diffusion_decoder = bool(getattr(config, "use_diffusion_decoder", False))
        self._train_scorer = bool(getattr(config, "train_scorer", True))
        self.score_decoder = ScoreDecoder(config)

        self._configure_trainable_parameters()

    def _count_enabled_sensors(self, keys: Tuple[str, ...]) -> int:
        return sum(1 for key in keys if len(self._config[key]) > 0)

    def _build_scene_backbones(self, config) -> None:
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(
                torch.randn(
                    1,
                    self.num_cams,
                    self._config.num_scene_tokens,
                    self.image_backbone.num_features,
                )
                * 1e-6,
                requires_grad=True,
            )

        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            lidar_num_features = getattr(
                self,
                "image_backbone",
                self.lidar_backbone,
            ).num_features
            self.lidar_scene_embeds = nn.Parameter(
                torch.randn(
                    1,
                    self.num_lidar,
                    self._config.num_scene_tokens,
                    lidar_num_features,
                )
                * 1e-6,
                requires_grad=True,
            )

    def _configure_trainable_parameters(self) -> None:
        train_diffusion_decoder_only = bool(
            getattr(self._config, "train_diffusion_decoder_only", False)
        )
        if not train_diffusion_decoder_only:
            self._set_score_decoder_trainable(self._train_scorer)
            return

        if not self._use_diffusion_decoder:
            log.warning(
                "train_diffusion_decoder_only=True but diffusion decoder is disabled; skip freeze."
            )
            self._set_score_decoder_trainable(self._train_scorer)
            return

        for p in self.parameters():
            p.requires_grad = False

        diffusion_decoder = self.trajectory_decoder.diffusion_decoder
        if diffusion_decoder is not None:
            for p in diffusion_decoder.parameters():
                p.requires_grad = True

        if self._train_scorer:
            self._set_score_decoder_trainable(True)

    def _set_score_decoder_trainable(self, trainable: bool) -> None:
        for p in self.score_decoder.parameters():
            p.requires_grad = trainable

    def scene_encoder(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ego_status_last = features["ego_status"][:, -1]
        if self._config.full_history_status:
            ego_status = features["ego_status"].flatten(-2)
        else:
            ego_status = ego_status_last
        log.debug(f"Ego Status - {ego_status.shape}")

        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")

        batch_size = ego_status.shape[0]
        scene_features: List[torch.Tensor] = []

        if self.num_cams > 0:
            if "image" in features:
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError("Expected image or camera_feature in features.")

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)
            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        encoded_scene = torch.cat(scene_features, dim=1)

        return {
            "ego_status_last": ego_status_last,
            "ego_status": ego_status,
            "ego_token": ego_token,
            "scene_features": encoded_scene,
        }


    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded_features = self.scene_encoder(features)
        output = self.trajectory_decoder(
            ego_token=encoded_features["ego_token"],
            scene_features=encoded_features["scene_features"],
            ego_status_last=encoded_features["ego_status_last"],
            vlm_hidden_state=features.get("vlm_hidden_state"),
            scenario_token=features.get("scenario_token"),
            targets=targets,
        )

        if self.training and not self._train_scorer:
            proposals = output["proposals"]
            output["trajectory"] = proposals[:, 0]
            output["pdm_score"] = torch.zeros(
                proposals.shape[0],
                proposals.shape[1],
                device=proposals.device,
                dtype=proposals.dtype,
            )
            output["pred_logit"] = None
            output["pred_logit2"] = None
            output["pred_agents_states"] = None
            output["pred_area_logit"] = None
            output["bev_semantic_map"] = None
            output["agent_states"] = None
            output["agent_labels"] = None
            return output

        score_output = self.score_decoder(
            proposals=output["proposals"],
            scene_features=encoded_features["scene_features"],
            ego_token=encoded_features["ego_token"],
            vlm_hidden_state=features.get("vlm_hidden_state"),
        )
        output.update(score_output)

        pdm_score = score_output["pdm_score"]
        selected_idx = torch.argmax(pdm_score, dim=1)
        output["trajectory"] = output["proposals"][
            torch.arange(pdm_score.shape[0], device=pdm_score.device),
            selected_idx,
        ]

        return output

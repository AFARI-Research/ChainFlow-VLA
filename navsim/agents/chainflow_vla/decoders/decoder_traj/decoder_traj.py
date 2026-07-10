from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from navsim.agents.chainflow_vla.utils import pylogger

from navsim.agents.chainflow_vla.layers.transformer import TransformerLayer
from navsim.agents.chainflow_vla.decoders.decoder_traj.ar import AutoregressiveTrajectoryDecoder
from navsim.agents.chainflow_vla.decoders.decoder_traj.dit import DiffusionTrajectoryDecoder


log = pylogger.get_pylogger(__name__)


class TrajectoryDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._poses_num = config.num_poses
        self._proposal_num = config.proposal_num
        self._use_diffusion_decoder = bool(getattr(config, "use_diffusion_decoder", False))

        self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)

        self.trajectory_transformer = TransformerLayer(
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
        )

        self.ar_decoders = nn.ModuleList(
            [AutoregressiveTrajectoryDecoder(config) for _ in range(config.ref_num + 1)]
        )
        self.diffusion_decoder: Optional[DiffusionTrajectoryDecoder]
        if self._use_diffusion_decoder:
            self.diffusion_decoder = DiffusionTrajectoryDecoder(config=config)
        else:
            self.diffusion_decoder = None


    def forward(
        self,
        ego_token: torch.Tensor,
        scene_features: torch.Tensor,
        ego_status_last: torch.Tensor,
        vlm_hidden_state: Optional[torch.Tensor] = None,
        scenario_token: Optional[Any] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        traj_tokens = ego_token + self.init_feature.weight[None]

        # initial trajectories
        proposals = self.ar_decoders[0](traj_tokens, scene_features, ego_status_last)
        proposal_list = [proposals]

        token_list = self.trajectory_transformer(traj_tokens, scene_features)
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.ar_decoders[i + 1](tokens, scene_features, ego_status_last)
            proposal_list.append(proposals)
        proposals = proposal_list[-1]
        
        if self._use_diffusion_decoder:
            diffusion_aux_list: list = []
            context_features = {
                "ar_proposals": proposals.detach(),
                "vlm_hidden_state": vlm_hidden_state,
                "ego_status_feature": ego_status_last[..., 3:],
                "scenario_token": scenario_token,
            }
            gt_trajectory = (targets or {}).get("trajectory") if self.training else None

            diff_result = self.diffusion_decoder(context_features, gt_trajectory)
            proposals = diff_result["proposals"]
            
            if self.training:
                diffusion_aux_list.append(
                    {
                        k: diff_result[k]
                        for k in ("pred_delta_normed", "target_delta_normed", "winner_idx")
                    }
                )

        output = {
            "proposals": proposals,
            "proposal_list": proposal_list,
        }
        if self._use_diffusion_decoder:
            output["diffusion_aux_list"] = diffusion_aux_list

        return output

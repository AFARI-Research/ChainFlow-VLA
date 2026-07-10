
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from lmdeploy import (
    ChatTemplateConfig,
    GenerationConfig,
    PytorchEngineConfig,
    pipeline,
)
from lmdeploy.vl import load_image as lmdeploy_load_image
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.chainflow_vla.encoders.vlm.utils.conversation import get_conv_template
from navsim.agents.chainflow_vla.encoders.vlm.utils.internvl_preprocess import (
    load_image as hf_internvl_load_image,
)
from navsim.common.dataclasses import (
    AgentInput,
    Scene,
    SensorConfig,
    Trajectory,
)
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


def format_number(n: float, decimal_places: int = 2) -> Union[float, str]:
    """Format for LMDeploy trajectory prompts (near-zero collapses to 0.0)."""
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    format_string = f"{{n:+.{decimal_places}f}}"
    return format_string.format(n=n)


def _format_number_hf_cache(n: float, decimal_places: int = 2) -> str:
    """String format for HF hidden-state caching prompts."""
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


class InternVLHFBackbone(nn.Module):
    """InternVL HF backbone for extracting hidden states (caching only)."""

    def __init__(self, model_type: str, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model = None
        self.tokenizer = None
        self.model_type = model_type.lower()
        self.device = device

        if self.model_type != "internvl":
            raise ValueError(
                f"Unsupported model_type: '{self.model_type}'. Only 'internvl' is supported."
            )

        logging.getLogger(__name__).info(
            "Initializing InternVL HF backbone from path: '%s'", checkpoint_path
        )

        self.model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
            device_map=self.device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self._configure_internvl()
        self.num_image_token = 256
        self.debug_token_layout = os.environ.get("VLM_DEBUG_TOKEN_LAYOUT", "0") == "1"

        logging.getLogger(__name__).info(
            "InternVL HF backbone loaded on device '%s'.", self.device
        )

    def _configure_internvl(self) -> None:
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id

    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int]):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized.")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question

            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(queries, return_tensors="pt", padding="max_length", max_length=2800)
        if self.debug_token_layout and len(queries) > 0:
            ids = model_inputs["input_ids"][0]
            mask = model_inputs["attention_mask"][0]
            pad_id = self.tokenizer.pad_token_id
            img_ctx_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            img_start_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
            img_end_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

            def _positions(token_id: int):
                return (ids == token_id).nonzero(as_tuple=True)[0]

            def _range(pos: torch.Tensor) -> str:
                if pos.numel() == 0:
                    return "[]"
                return f"[{int(pos[0])}, {int(pos[-1])}]"

            pad_pos = _positions(pad_id) if pad_id is not None else torch.empty(0, dtype=torch.long)
            img_ctx_pos = _positions(img_ctx_id)
            img_start_pos = _positions(img_start_id)
            img_end_pos = _positions(img_end_id)
            text_pos = (
                (mask == 1)
                & (ids != img_ctx_id)
                & (ids != img_start_id)
                & (ids != img_end_id)
            ).nonzero(as_tuple=True)[0]

            logging.getLogger(__name__).info(
                "[token-debug] seq_len=%d real_tokens=%d pad_tokens=%d image_ctx_tokens=%d text_tokens=%d",
                ids.numel(),
                int(mask.sum().item()),
                int(pad_pos.numel()),
                int(img_ctx_pos.numel()),
                int(text_pos.numel()),
            )
            logging.getLogger(__name__).info(
                "[token-debug] pad_range=%s image_ctx_range=%s image_start=%s image_end=%s text_range=%s",
                _range(pad_pos),
                _range(img_ctx_pos),
                _range(img_start_pos),
                _range(img_end_pos),
                _range(text_pos),
            )
        device = torch.device("cuda")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

        return self.model(
            pixel_values=pixel_values.bfloat16(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags.squeeze(-1),
            output_hidden_states=True,
            return_dict=True,
        )


class InternVLHiddenStateFeatureBuilder(AbstractFeatureBuilder):
    """Builds InternVL HF features including `last_hidden_state` for offline caching."""

    def __init__(
        self,
        cache_hidden_state: bool = True,
        model_type: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        cache_mode: bool = False,
        dynamic_image_size: bool = True,
        use_thumbnail: bool = True,
        force_image_size: int = 448,
        max_dynamic_patch: int = 16,
    ):
        super().__init__()
        self.cache_hidden_state = cache_hidden_state
        self.backbone: Optional[InternVLHFBackbone] = None
        self.cache_mode = cache_mode
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.force_image_size = force_image_size
        self.max_dynamic_patch = max_dynamic_patch

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError(
                    "When cache_hidden_state=True and cache_mode=True, "
                    "`model_type` and `checkpoint_path` must be provided."
                )
            self.backbone = InternVLHFBackbone(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device=device,
            )

    def _load_image_for_vlm(self, image_path: str) -> torch.Tensor:
        if self.dynamic_image_size:
            return hf_internvl_load_image(
                image_path,
                input_size=self.force_image_size,
                max_num=self.max_dynamic_patch,
                use_thumbnail=self.use_thumbnail,
            )
        return hf_internvl_load_image(
            image_path,
            input_size=self.force_image_size,
            max_num=1,
            use_thumbnail=False,
        )

    def get_unique_name(self) -> str:
        return "internvl_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32,
        )
        high_command_one_hot = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        status_feature = torch.cat(
            [
                high_command_one_hot.clone(),
                torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
            dim=-1,
        )

        if not self.cache_hidden_state:
            image_path = str(cameras[-1].cam_f0.image)
            path_as_ordinals = [ord(char) for char in image_path]
            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)
            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "status_feature": status_feature.cpu(),
                "image_path_tensor": path_tensor.cpu(),
            }

        if self.backbone is None:
            raise RuntimeError("InternVLHiddenStateFeatureBuilder: backbone not initialized.")

        pixel_values = self._load_image_for_vlm(str(cameras[-1].cam_f0.image)).unsqueeze(0)
        pixel_values_squeezed = pixel_values.squeeze(1)
        num_patches_list = [pv.shape[0] for pv in pixel_values_squeezed]
        pixel_values_cat = torch.cat(list(pixel_values_squeezed), dim=0)

        navigation_commands = ["turn left", "go straight", "turn right"]
        command_str = next(
            (navigation_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1),
            "unknown",
        )
        history_str = " ".join(
            [
                f"   - t-{3-i}: ({_format_number_hf_cache(history_trajectory[i, 0].item())}, "
                f"{_format_number_hf_cache(history_trajectory[i, 1].item())}, "
                f"{_format_number_hf_cache(history_trajectory[i, 2].item())})"
                for i in range(4)
            ]
        )

        prompt = (
            f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
            f"1. Visual perception from front camera view\n"
            f"2. Historical motion context (last 4 timesteps):{history_str}\n"
            f"3. Active navigation command: [{command_str.upper()}]"
        )
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        questions = [f"{prompt}{output_requirements}"]

        outputs = self.backbone(pixel_values_cat.cuda(), questions, num_patches_list=num_patches_list)
        last_hidden_state = outputs.hidden_states[-1]

        return {
            "history_trajectory": history_trajectory.cpu(),
            "high_command_one_hot": high_command_one_hot.cpu(),
            "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
            "status_feature": status_feature.cpu(),
        }

class InternVLFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder for InternVLAgent.

    - Default: returns raw `ego_statuses` and `cameras` for LMDeploy inference.
    - Cache mode (`cache_hidden_state=True` and `cache_mode=True`): uses
      `InternVLHiddenStateFeatureBuilder` for offline hidden-state caching.
    """

    def __init__(
        self,
        cache_hidden_state: bool = False,
        cache_mode: bool = False,
        checkpoint_path: Optional[str] = None,
        cam_type: str = "single",
        dynamic_image_size: bool = True,
        use_thumbnail: bool = True,
        force_image_size: int = 448,
        max_dynamic_patch: int = 12,
    ):
        self.cache_hidden_state = cache_hidden_state
        self.cache_mode = cache_mode
        self.cam_type = cam_type
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.force_image_size = force_image_size
        self.max_dynamic_patch = max_dynamic_patch
        self._hf_cache_builder: Optional[InternVLHiddenStateFeatureBuilder] = None

        if self.cache_hidden_state and self.cache_mode:
            if not checkpoint_path:
                raise ValueError(
                    "When cache_hidden_state=True and cache_mode=True, "
                    "checkpoint_path must be provided for InternVLAgent."
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._hf_cache_builder = InternVLHiddenStateFeatureBuilder(
                cache_hidden_state=True,
                model_type="internvl",
                checkpoint_path=checkpoint_path,
                device=device,
                cache_mode=True,
                dynamic_image_size=dynamic_image_size,
                use_thumbnail=use_thumbnail,
                force_image_size=force_image_size,
                max_dynamic_patch=max_dynamic_patch,
            )

    def get_unique_name(self) -> str:
        return "internvl_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        if not (self.cache_hidden_state and self.cache_mode):
            ego_statuses = agent_input.ego_statuses
            cameras = agent_input.cameras
            return {"ego_statuses": ego_statuses, "cameras": cameras}

        if self._hf_cache_builder is None:
            raise RuntimeError(
                "InternVLFeatureBuilder is in cache_hidden_state mode, "
                "but InternVLHiddenStateFeatureBuilder was not initialized."
            )
        return self._hf_cache_builder.compute_features(agent_input)

class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Target builder for trajectory supervision."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class InternVLAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: Optional[str] = None,
        prompt_type: Optional[str] = "base",
        cam_type: Optional[str] = "single",
        cache_mode: bool = False,
        cache_hidden_state: bool = False,
        dynamic_image_size: bool = True,
        use_thumbnail: bool = True,
        force_image_size: int = 448,
        max_dynamic_patch: int = 16,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.checkpoint_path = checkpoint_path
        self.prompt_type = prompt_type
        self.cam_type = cam_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.force_image_size = force_image_size
        self.max_dynamic_patch = max_dynamic_patch

        self.pipe = None
        if not (self.cache_hidden_state and self.cache_mode):
            self.pipe = pipeline(
                self.checkpoint_path,
                backend_config=PytorchEngineConfig(session_len=8192, dtype="bfloat16"),
                chat_template_config=ChatTemplateConfig(
                    model_name="internvl2_5", meta_instruction=system_message
                ),
            )

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [
            InternVLFeatureBuilder(
                cache_hidden_state=self.cache_hidden_state,
                cache_mode=self.cache_mode,
                checkpoint_path=self.checkpoint_path,
                cam_type=self.cam_type,
                dynamic_image_size=self.dynamic_image_size,
                use_thumbnail=self.use_thumbnail,
                force_image_size=self.force_image_size,
                max_dynamic_patch=self.max_dynamic_patch,
            )
        ]

    def forward(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, torch.Tensor]:
        if self.pipe is None:
            raise RuntimeError(
                "InternVLAgent was constructed with cache_hidden_state=True and cache_mode=True; "
                "LMDeploy pipeline is disabled in this mode and forward() should not be used. "
                "Use this configuration only for offline caching."
            )
        ego_statuses = features["ego_statuses"]
        cameras = features["cameras"]

        history_trajectory = []
        for i in range(4):
            ego_status = ego_statuses[i]
            history_trajectory.append(
                {
                    "x": format_number(ego_status.ego_pose[0]),
                    "y": format_number(ego_status.ego_pose[1]),
                    "heading": format_number(ego_status.ego_pose[2]),
                }
            )

        high_command_one_hot = ego_statuses[-1].driving_command
        navigation_commands = ["turn left", "go straight", "turn right"]
        command_str = [
            navigation_commands[i] for i in range(len(high_command_one_hot)) if high_command_one_hot[i] == 1
        ]
        command_str = command_str[0] if command_str else "unknown"

        image_paths = []
        image_prompt_lines = []
        image_prompt_desc = ""

        if self.cam_type == "single":
            image_paths.append(str(cameras[-1].cam_f0.image))
            image_prompt_lines.append("<FRONT VIEW>:\n<image>\n")
            image_prompt_desc = "1. Visual perception from front camera view\n"
        elif self.cam_type == "multi_view":
            image_paths.extend(
                [
                    str(cameras[-1].cam_f0.image),
                    str(cameras[-1].cam_l0.image),
                    str(cameras[-1].cam_r0.image),
                    str(cameras[-1].cam_l2.image),
                    str(cameras[-1].cam_r2.image),
                    str(cameras[-1].cam_b0.image),
                ]
            )
            image_prompt_lines.append(
                "<FRONT VIEW>:\n<image>\n<FRONT LEFT VIEW>:\n<image>\n<FRONT RIGHT VIEW>:\n<image>\n"
                "<BACK LEFT VIEW>:\n<image>\n<BACK RIGHT VIEW>:\n<image>\n<BACK VIEW>:\n<image>\n"
            )
            image_prompt_desc = "1. Visual perception from the six surrounding camera views\n"
        elif self.cam_type == "cont":
            for i in range(4):
                image_paths.append(str(cameras[i].cam_f0.image))
                image_prompt_lines.append(f"<FRONT VIEW>Frame-{i+1}: <image>\n")
            image_prompt_desc = (
                "1. Visual perception from continuous front camera views of the last 4 timesteps\n"
            )

        pixel_values = [lmdeploy_load_image(image_path) for image_path in image_paths]

        generation_config = GenerationConfig(
            max_new_tokens=512,
            min_new_tokens=50,
            do_sample=False,
            temperature=0.0,
        )

        image_prompt_str = "".join(image_prompt_lines)

        common_prompt = (
            f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt_desc}"""
            f"""2. Historical motion context (last 4 timesteps):{" ".join([f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n"""
            f"3. Active navigation command: [{command_str.upper()}]"
        )

        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )

        if self.prompt_type == "vel_and_acc":
            current_ego_status = ego_statuses[-1]
            vel_acc_info = (
                f"\n4. Current velocity: ({format_number(current_ego_status.ego_velocity[0])}, "
                f"{format_number(current_ego_status.ego_velocity[1])})"
                f"\n5. Current acceleration: ({format_number(current_ego_status.ego_acceleration[0])}, "
                f"{format_number(current_ego_status.ego_acceleration[1])})"
            )
            question = f"{image_prompt_str}\n{common_prompt}{vel_acc_info}{output_requirements}"
        else:
            question = f"{''.join(['<image>' for _ in range(len(image_paths))])}\n{common_prompt}{output_requirements}"

        prompts = [(question, pixel_values)]

        responses = self.pipe(prompts, gen_config=generation_config)
        answers = [response.text for response in responses]

        full_match = re.search(
            r"\[PT(?:, )?((?:\([-+]?\d*\.\d+, [-+]?\d*\.\d+, [-+]?\d*\.\d+\)(?:, )?){8})\]",
            answers[0],
        )
        if full_match:
            coords_matches = re.findall(
                r"\(([-+]?\d*\.\d+), ([-+]?\d*\.\d+), ([-+]?\d*\.\d+)\)", full_match.group(1)
            )
            if len(coords_matches) == 8:
                coordinates = [tuple(map(float, coord)) for coord in coords_matches]
                coordinates_array = np.array(coordinates, dtype=np.float32)
                return {"trajectory": coordinates_array.reshape(1, self._trajectory_sampling.num_poses, 3)}

        logging.getLogger(__name__).warning(
            "Error parsing trajectory, returning zeros: %s", answers[0] if answers else answers
        )
        return {"trajectory": np.zeros((1, self._trajectory_sampling.num_poses, 3), dtype=np.float32)}

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0)

        return Trajectory(poses)

    def compute_loss(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)

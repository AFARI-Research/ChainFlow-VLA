# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAVSIM (https://github.com/valeoai/DrivoR)
# Copyright (c) Valeoai-Model. All Rights Reserved.
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict
import numpy as np

import torch

from navsim.common.enums import LidarIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
)

class ChainFlowVLAFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: Dict):
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "chainflow_vla_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        features = {}
        data_camera = self._get_camera_feature(agent_input)
        features.update(data_camera)


        if len(self._config.lidar_pc) > 0:
            data_lidar = self._get_lidar_feature(agent_input)
            features.update(data_lidar)

        ego_feature_list=[]

        for ego_status in agent_input.ego_statuses:
            if ego_status is None:
                continue
            pose=torch.tensor(ego_status.ego_pose, dtype=torch.float32)
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
            driving_command = torch.tensor(ego_status.driving_command, dtype=torch.float32)
            ego_feature=torch.cat([pose,velocity, acceleration, driving_command], dim=-1)

            ego_feature_list.append(ego_feature)

        features["ego_status"] =torch.stack(ego_feature_list)

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        from PIL import Image

        cameras = agent_input.cameras[-1]

        # cameras = [cameras.cam_b0, cameras.cam_f0, cameras.cam_l0, cameras.cam_l1, cameras.cam_l2, cameras.cam_r0, cameras.cam_r1, cameras.cam_r2]

        # this is a change for the focus front cam
        cameras = [cameras.cam_f0, cameras.cam_b0, cameras.cam_l0, cameras.cam_l1, cameras.cam_l2, cameras.cam_r0, cameras.cam_r1, cameras.cam_r2]

        images = []
        cam_Ks = []
        lidar2cams = []
        for cam in cameras:
            if cam.image is None:
                continue

            im = Image.fromarray(cam.image)
            cam_K = np.array(cam.intrinsics)
            sensor2lidar_rotation = np.asarray(cam.sensor2lidar_rotation)
            sensor2lidar_translation = np.asarray(cam.sensor2lidar_translation)
            sensor2lidar_rt = np.eye(4)
            sensor2lidar_rt[:3, :3] = sensor2lidar_rotation
            sensor2lidar_rt[:3, 3] = sensor2lidar_translation
            lidar2cam_rt = np.linalg.inv(sensor2lidar_rt)

            # intrinsics resize
            original_size = im.size
            cam_K = cam_K.clone() if isinstance(cam_K, torch.Tensor) else cam_K.copy() # torch.Size([8, 3, 3])
            cam_K[0, 0] = cam_K[0, 0] * self._config.image_size[0] / original_size[0]
            cam_K[1, 1] = cam_K[1, 1] * self._config.image_size[1] / original_size[1]
            cam_K[0, 2] = cam_K[0, 2] * self._config.image_size[0] / original_size[0]
            cam_K[1, 2] = cam_K[1, 2] * self._config.image_size[1] / original_size[1]

            # image resize
            im = im.resize(self._config.image_size)

            # PIL to numpy and normalize
            im = np.asarray(im, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            im = (im - mean) / std

            # convert to torch
            im = torch.from_numpy(im).permute(2, 0, 1)
            cam_K = torch.from_numpy(cam_K)
            lidar2cam_rt = torch.from_numpy(lidar2cam_rt)

            images.append(im)
            cam_Ks.append(cam_K)
            lidar2cams.append(lidar2cam_rt)


        # Collect all camera images in a list for easier processing
        data = {
            "image": torch.stack(images),
            "cam_K": torch.stack(cam_Ks),
            "world_2_cam": torch.stack(lidar2cams)
        }

        # raise NotImplementedError


        # data["image"] = torch.stack([transforms.ToTensor()(img) for img in data["image"]])
        # data["cam_K"] = torch.stack([torch.from_numpy(cam) for cam in data["cam_K"]])
        # data["world_2_cam"] = torch.stack([torch.from_numpy(world_2_cam) for world_2_cam in data["world_2_cam"]])

        # data["image"] = data["image"].unsqueeze(0) # add time dimension
        # data["cam_K"] = data["cam_K"].unsqueeze(0) # add time dimension
        # data["world_2_cam"] = data["world_2_cam"].unsqueeze(0) # add time dimension

        return data


    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # # only consider (x,y,z) & swap axes for (N,3) numpy array
        # lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # lidar_feature = voxelize_with_feature_averaging(lidar_pc, grid_dims=self._config.grid_dims, grid_range=self._config.grid_range)

        # return {"lidar_feature": lidar_feature}

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                self._config.lidar_image_size[0]+1,
                # (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                self._config.lidar_image_size[1]+1,
                # (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.lidar_hist_max_per_pixel] = self._config.lidar_hist_max_per_pixel
            overhead_splat = hist / self._config.lidar_hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.lidar_max_height]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.lidar_use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        features = np.expand_dims(features, axis=0) # add a dimension for the number of sensors (actually 1s)

        # return torch.tensor(features)
        return {"lidar_feature": torch.tensor(features)}

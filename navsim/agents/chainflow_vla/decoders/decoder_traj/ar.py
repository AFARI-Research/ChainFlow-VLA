# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# -----------------------------------------------------------------------

from typing import Optional, Tuple

import torch
import torch.nn as nn
from navsim.agents.chainflow_vla.utils.utils import (
    SinCosTimePosEmb,
    StateEmbed,
    bicycle_integrate_xyh_v,
)


class AutoregressiveTrajectoryDecoder(nn.Module):
    """
    Latent-autoregressive trajectory head.

    Instead of expanding one static proposal token to all steps, this head keeps
    a per-proposal latent state h_t. Each step predicts motion using h_t,
    previous rollout state, and scene features, then updates h_t -> h_{t+1}.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self._num_poses = int(config.num_poses)
        self._proposal_num = int(config.proposal_num)
        self._d_model = int(config.tf_d_model)
        self._d_ffn = int(config.tf_d_ffn)

        self._one_token_per_traj = bool(getattr(config, "one_token_per_traj", False))
        self._tf_num_head = int(getattr(config, "tf_num_head", 8))
        self._cfg_dropout = float(getattr(config, "tf_dropout", 0.0))
        self._tf_dropout = self._cfg_dropout if self._cfg_dropout > 0.0 else 0.1
        self._ar_num_layers = int(getattr(config, "ar_decoder_num_layers", 1))
        self._use_bicycle = bool(getattr(config, "ar_use_bicycle_kinematics", False))
        self._bicycle_dt = float(getattr(config, "ar_bicycle_dt", 0.5))

        self.time_pos = SinCosTimePosEmb(dim=self._d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_model),
            nn.SiLU(),
            nn.Linear(self._d_model, self._d_model),
        )
        self.state_embed = StateEmbed(self._d_model, xy_scale=1.0)

        self.cross_attn = nn.MultiheadAttention(
            self._d_model,
            self._tf_num_head,
            dropout=self._tf_dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self._d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.GELU(),
            nn.Dropout(self._tf_dropout),
            nn.Linear(self._d_ffn, self._d_model),
        )
        self.ffn_norm = nn.LayerNorm(self._d_model)        

        self.hidden_update = nn.GRUCell(self._d_model, self._d_model)
        self.hidden_norm = nn.LayerNorm(self._d_model)

        self.delta_head = nn.Sequential(
            nn.Linear(self._d_model, self._d_model),
            nn.GELU(),
            nn.Linear(self._d_model, 2),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.delta_head[-1].weight, gain=0.01)
        nn.init.constant_(self.delta_head[-1].bias, 0.0)

    def _run_step_blocks(self, query_t: torch.Tensor, scene_features: torch.Tensor) -> torch.Tensor:
        cross_out, _ = self.cross_attn(query_t, scene_features, scene_features)
        query_t = self.cross_norm(query_t + cross_out)
        query_t = self.ffn_norm(query_t + self.ffn(query_t))

        return query_t

    def _update_hidden(self, traj_tokens: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
        batch_size, proposal_num, _ = traj_tokens.shape
        next_hidden = self.hidden_update(
            query_t.reshape(batch_size * proposal_num, self._d_model),
            traj_tokens.reshape(batch_size * proposal_num, self._d_model),
        )
        next_hidden = next_hidden.reshape(batch_size, proposal_num, self._d_model)
        return self.hidden_norm(next_hidden)

    def forward(
        self,
        traj_tokens: torch.Tensor,
        scene_features: torch.Tensor,
        ego_status_last: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, proposal_num, _ = traj_tokens.shape
        device = traj_tokens.device
        dtype = traj_tokens.dtype

        prev_xyh = torch.zeros(batch_size, proposal_num, 3, device=device, dtype=dtype)

        if ego_status_last is not None:
            vxvy = ego_status_last[:, 3:5].to(device=device, dtype=dtype)
            v0 = torch.linalg.norm(vxvy, dim=-1).clamp(min=1e-4)
        else:
            v0 = torch.zeros(batch_size, device=device, dtype=dtype) + 1e-4
        prev_v = v0[:, None].expand(-1, proposal_num).contiguous()

        time_steps = torch.arange(self._num_poses, device=device, dtype=torch.long)
        time_emb_all = self.time_mlp(self.time_pos(time_steps))

        states_list = []
        for t in range(self._num_poses):
            query_t = traj_tokens + time_emb_all[t].view(1, 1, self._d_model)
            query_t = query_t + self.state_embed(prev_xyh.detach())
            query_t = self._run_step_blocks(query_t, scene_features)
            raw = self.delta_head(query_t).clamp(min=-1e2, max=1e2)

            accel_t = raw[..., 0]
            yaw_rate_t = raw[..., 1]
            state_t, prev_v = bicycle_integrate_xyh_v(
                prev_xyh, prev_v, accel_t, yaw_rate_t, self._bicycle_dt
            )

            states_list.append(state_t)
            traj_tokens = self._update_hidden(traj_tokens, query_t)
            prev_xyh = state_t

        return torch.stack(states_list, dim=2)

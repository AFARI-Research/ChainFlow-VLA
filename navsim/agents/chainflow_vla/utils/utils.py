# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.chainflow_vla.utils.embeddings import SinusoidalTimestepEmbedding


def bicycle_integrate_xyh_v(
    xyh: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    yaw_rate: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = xyh[..., 0]
    y = xyh[..., 1]
    theta = xyh[..., 2]
    v_next = (v + a * dt).clamp(min=0.0)
    theta_raw = theta + yaw_rate * dt
    theta_next = torch.atan2(torch.sin(theta_raw), torch.cos(theta_raw))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    cos_n, sin_n = torch.cos(theta_next), torch.sin(theta_next)
    x_next = x + 0.5 * (v * cos_t + v_next * cos_n) * dt
    y_next = y + 0.5 * (v * sin_t + v_next * sin_n) * dt
    xyh_next = torch.stack([x_next, y_next, theta_next], dim=-1)
    return xyh_next, v_next


class SinCosTimePosEmb(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = float(max_period)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period)
            * torch.arange(0, half, device=device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps.to(dtype=torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat(
                [emb, torch.zeros(*emb.shape[:-1], 1, device=device, dtype=emb.dtype)],
                dim=-1,
            )
        return emb


class StateEmbed(nn.Module):
    def __init__(self, d_model: int, xy_scale: float = 0.05) -> None:
        super().__init__()
        self.xy_scale = xy_scale
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, xyh: torch.Tensor) -> torch.Tensor:
        x, y, h = xyh[..., 0:1], xyh[..., 1:2], xyh[..., 2:3]
        xyh_scaled = torch.cat(
            [x * self.xy_scale, y * self.xy_scale, torch.atan2(torch.sin(h), torch.cos(h))],
            dim=-1,
        )
        return self.mlp(xyh_scaled.float())


class WaypointActionEncoder(nn.Module):
    """Fuses per-waypoint action deltas with sinusoidal timestep embeddings (diffusion decoder input)."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.time_embed = SinusoidalTimestepEmbedding(hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.dim() != 1 or timesteps.shape[0] != batch_size:
            raise ValueError(
                f"Expected timesteps shape ({batch_size},), got {tuple(timesteps.shape)}."
            )

        action_embedding = self.action_proj(actions.float())
        timesteps = timesteps[:, None].expand(-1, horizon).reshape(-1)
        time_embedding = self.time_embed(timesteps).reshape(batch_size, horizon, -1)
        time_embedding = time_embedding.to(device=actions.device, dtype=action_embedding.dtype)
        x = torch.cat((action_embedding, time_embedding), dim=-1)
        return self.fc2(self.act(self.fc1(x)))


class ARProposalEncoder(nn.Module):
    """Encode clean AR proposal waypoints as diffusion proposal tokens."""

    def __init__(self, action_dim: int, hidden_size: int, horizon: int):
        super().__init__()
        self.proposal_proj = nn.Linear(action_dim, hidden_size)
        self.waypoint_embed = nn.Parameter(torch.zeros(1, horizon, hidden_size))
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, proposal: torch.Tensor) -> torch.Tensor:
        _, horizon, _ = proposal.shape
        x = self.proposal_proj(proposal.float())
        x = x + self.waypoint_embed[:, :horizon, :].to(device=x.device, dtype=x.dtype)
        x = x + self.mlp(x)
        return self.out_norm(x)


class _ARCrossFFNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        dropout: float,
        use_mode_self_attn: bool = False,
    ) -> None:
        super().__init__()
        if use_mode_self_attn:
            self.mode_self_attn = nn.MultiheadAttention(
                d_model,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.mode_self_norm = nn.LayerNorm(d_model)
        else:
            self.mode_self_attn = None
            self.mode_self_norm = None
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_t: torch.Tensor,
        scene_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode_self_attn is not None:
            mode_out, _ = self.mode_self_attn(query_t, query_t, query_t)
            query_t = self.mode_self_norm(query_t + mode_out)
        cross_out, _ = self.cross_attn(query_t, scene_features, scene_features)
        query_t = self.cross_norm(query_t + cross_out)
        query_t = self.ffn_norm(query_t + self.ffn(query_t))
        return query_t

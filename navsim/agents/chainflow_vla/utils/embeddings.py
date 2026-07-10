# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE cache for waypoint-token attention."""

    def __init__(self, dim: int, max_position_embeddings: int, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}.")

        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device) -> None:
        self.max_seq_len_cached = seq_len
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(position_ids.max().item()) + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)

        batch_size = x.shape[0]
        if position_ids.shape[0] == 1 and batch_size != 1:
            position_ids = position_ids.expand(batch_size, -1)

        gather_ids = position_ids.unsqueeze(1).unsqueeze(3).expand(-1, 1, -1, self.dim)
        cos_cache = self.cos_cached.to(device=x.device).expand(position_ids.shape[0], -1, -1, -1)
        sin_cache = self.sin_cached.to(device=x.device).expand(position_ids.shape[0], -1, -1, -1)
        cos = cos_cache.gather(2, gather_ids)
        sin = sin_cache.gather(2, gather_ids)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

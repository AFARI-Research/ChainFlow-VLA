# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import secrets
from typing import Any, Optional

import torch
from torch.distributions import Beta


class FlowMatchingTimeSampler:
    def __init__(
        self,
        num_timestep_buckets: int,
        noise_scale: float,
        beta_alpha: float,
        beta_beta: float,
    ):
        if noise_scale <= 0.0:
            raise ValueError(f"flow_matching_noise_scale must be positive, got {noise_scale}.")
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_scale = noise_scale
        self.beta_dist = Beta(beta_alpha, beta_beta)

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample(torch.Size([batch_size])).to(device=device, dtype=dtype)
        return (self.noise_scale - sample) / self.noise_scale

    def to_discrete_timesteps(self, t_cont: torch.Tensor) -> torch.Tensor:
        return (t_cont * self.num_timestep_buckets).long()


def stable_seed(token: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{token}:{base_seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def initial_inference_noise(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    eval_noise_seed: Optional[int],
    scenario_tokens: Optional[Any],
    training: bool,
) -> torch.Tensor:
    if training:
        return torch.randn(shape, device=device, dtype=dtype)

    if eval_noise_seed is None or scenario_tokens is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(secrets.randbits(63))
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    batch_size = shape[0]
    tokens = list(scenario_tokens)
    if len(tokens) != batch_size:
        raise ValueError(
            "scenario_token batch size mismatch for deterministic diffusion noise: "
            f"got {len(tokens)}, expected {batch_size}."
        )

    sample_shape = shape[1:]
    samples = []
    for token in tokens:
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(str(token), eval_noise_seed))
        samples.append(torch.randn(sample_shape, device=device, dtype=dtype, generator=generator))
    return torch.stack(samples, dim=0)

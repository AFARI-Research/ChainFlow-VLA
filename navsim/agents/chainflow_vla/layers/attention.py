# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.chainflow_vla.utils.embeddings import rotate_half


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.to(dtype=x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        qk_norm: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        if dim_head is None:
            if query_dim % heads != 0:
                raise ValueError(f"query_dim={query_dim} must be divisible by heads={heads}")
            dim_head = query_dim // heads

        self.inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_dim = dim_head
        self.scale = dim_head**-0.5

        context_dim = cross_attention_dim or query_dim
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_embedder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        batch_size, query_tokens, _ = hidden_states.shape
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        is_self_attention = encoder_hidden_states is None

        q = (
            self.to_q(hidden_states)
            .view(batch_size, query_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.to_k(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rotary_embedder is not None:
            position_ids = torch.arange(query_tokens, device=hidden_states.device).unsqueeze(0)
            cos, sin = rotary_embedder(hidden_states, position_ids)
            q = (q * cos) + (rotate_half(q) * sin)
            if is_self_attention:
                k = (k * cos) + (rotate_half(k) * sin)

        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.to_out[-1].p if self.training else 0.0,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.to_out[-1].p, training=self.training)
            x = torch.matmul(attn_probs, v)

        x = x.transpose(1, 2).reshape(batch_size, query_tokens, self.inner_dim)
        return self.to_out(x)

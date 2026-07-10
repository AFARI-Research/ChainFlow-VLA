from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from navsim.agents.chainflow_vla.layers.attention import Attention
from navsim.agents.chainflow_vla.layers.timm_layers import Mlp
from navsim.agents.chainflow_vla.utils import pylogger
from navsim.agents.chainflow_vla.utils.embeddings import RotaryEmbedding, SinusoidalTimestepEmbedding
from navsim.agents.chainflow_vla.utils.sampling import FlowMatchingTimeSampler, initial_inference_noise
from navsim.agents.chainflow_vla.utils.utils import ARProposalEncoder, WaypointActionEncoder

log = pylogger.get_pylogger(__name__)


class DiTBlock(nn.Module):
    """Single DiT block with adaLN-Zero self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.0,
        attention_mode: str = "full",
    ):
        super().__init__()
        if attention_mode not in {"full", "self", "cross"}:
            raise ValueError(f"Unsupported attention_mode: {attention_mode}.")

        self.has_self_attn = attention_mode in {"full", "self"}
        self.has_cross_attn = attention_mode in {"full", "cross"}

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        if self.has_self_attn:
            self.self_attn = Attention(query_dim=d_model, heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        if self.has_cross_attn:
            self.cross_attn = Attention(query_dim=d_model, heads=num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        conditioning: torch.Tensor,
        rotary_embedder: Optional[nn.Module] = None,
        query_token_count: Optional[int] = None,
    ) -> torch.Tensor:
        mod_params = self.adaLN_modulation(conditioning)
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = mod_params.chunk(9, dim=-1)

        shift1, scale1, gate1 = shift1.unsqueeze(1), scale1.unsqueeze(1), gate1.unsqueeze(1)
        shift2, scale2, gate2 = shift2.unsqueeze(1), scale2.unsqueeze(1), gate2.unsqueeze(1)
        shift3, scale3, gate3 = shift3.unsqueeze(1), scale3.unsqueeze(1), gate3.unsqueeze(1)

        if self.has_self_attn:
            h = self.modulate(self.norm1(x), shift1, scale1)
            h = self.self_attn(h, rotary_embedder=rotary_embedder)
            x = x + gate1 * h

        if self.has_cross_attn:
            if query_token_count is None or query_token_count >= x.shape[1]:
                h = self.modulate(self.norm2(x), shift2, scale2)
                h = self.cross_attn(h, encoder_hidden_states=cond_tokens, rotary_embedder=rotary_embedder)
                x = x + gate2 * h
            else:
                x_query = x[:, :query_token_count]
                h = self.modulate(self.norm2(x_query), shift2, scale2)
                h = self.cross_attn(h, encoder_hidden_states=cond_tokens, rotary_embedder=rotary_embedder)
                x = torch.cat((x_query + gate2 * h, x[:, query_token_count:]), dim=1)

        h = self.modulate(self.norm3(x), shift3, scale3)
        return x + gate3 * self.ffn(h)


class DiffusionTrajectoryDecoder(nn.Module):
    """DiT-based diffusion trajectory decoder operating in AR proposal delta space."""

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._num_poses = int(config.num_poses)
        self._d_model = int(config.tf_d_model)
        self._d_ffn = int(config.tf_d_ffn)

        self._diffusion_type = str(getattr(config, "diffusion_type", "ddim")).lower()
        if self._diffusion_type not in {"ddim", "flow_matching"}:
            raise ValueError(
                f"Unsupported diffusion_type: {self._diffusion_type}. "
                "Expected 'ddim' or 'flow_matching'."
            )

        if self._diffusion_type == "ddim":
            try:
                from diffusers.schedulers import DDIMScheduler
            except ImportError as exc:
                raise ImportError(
                    "diffusers is required when diffusion_type='ddim'. "
                    "Install project dependencies or use diffusion_type='flow_matching'."
                ) from exc
            self._num_train_timesteps = int(getattr(config, "ddim_num_train_timesteps", 1000))
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self._num_train_timesteps,
                beta_schedule=str(getattr(config, "ddim_beta_schedule", "scaled_linear")),
                prediction_type="epsilon",
                clip_sample=False,
            )
        else:
            self._num_train_timesteps = None
            self.noise_scheduler = None

        self._num_inference_steps = int(getattr(config, "diffusion_num_inference_steps", 10))
        self._flow_time_sampler = FlowMatchingTimeSampler(
            num_timestep_buckets=int(getattr(config, "flow_matching_num_timestep_buckets", 1000)),
            noise_scale=float(getattr(config, "flow_matching_noise_scale", 0.999)),
            beta_alpha=float(getattr(config, "flow_matching_noise_beta_alpha", 1.5)),
            beta_beta=float(getattr(config, "flow_matching_noise_beta_beta", 1.0)),
        )
        self._delta_scale = float(getattr(config, "diffusion_delta_scale", 1.0))
        eval_noise_seed = getattr(config, "diffusion_eval_noise_seed", None)
        if eval_noise_seed is None or str(eval_noise_seed).lower() in {"", "none", "null"}:
            self._eval_noise_seed = None
        else:
            self._eval_noise_seed = int(eval_noise_seed)

        self._use_ar_proposal_tokens = bool(getattr(config, "diffusion_use_ar_proposal_tokens", False))

        self.action_encoder = WaypointActionEncoder(action_dim=3, hidden_size=self._d_model)
        self.traj_output_proj = nn.Linear(self._d_model, 3)
        nn.init.zeros_(self.traj_output_proj.weight)
        nn.init.zeros_(self.traj_output_proj.bias)

        vlm_hidden_dim = int(getattr(config, "diffusion_vlm_hidden_dim", 1536))
        self.vlm_feature_encoder = nn.Linear(vlm_hidden_dim, self._d_model)
        self.init_fusion_projector = nn.Linear(self._d_model * 2, self._d_model)
        if self._use_ar_proposal_tokens:
            self.ar_proposal_encoder = ARProposalEncoder(
                action_dim=3,
                hidden_size=self._d_model,
                horizon=self._num_poses,
            )
            self.ar_proposal_token_type_embed = nn.Parameter(torch.empty(2, self._d_model))
            nn.init.normal_(self.ar_proposal_token_type_embed, mean=0.0, std=0.02)

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(self._d_model),
            nn.Linear(self._d_model, self._d_model * 4),
            nn.GELU(),
            nn.Linear(self._d_model * 4, self._d_model),
        )

        ego_status_dim = int(getattr(config, "diffusion_ego_status_dim", 11))
        self.ego_status_encoder = Mlp(
            in_features=ego_status_dim,
            hidden_features=self._d_ffn,
            out_features=self._d_model,
            norm_layer=nn.LayerNorm,
        )

        num_layers = int(getattr(config, "diffusion_num_layers", 4))
        num_heads = int(getattr(config, "diffusion_num_heads", 8))
        if self._d_model % num_heads != 0:
            raise ValueError(
                f"tf_d_model={self._d_model} must be divisible by diffusion_num_heads={num_heads}."
            )
        head_dim = self._d_model // num_heads
        self.position_embedding = nn.Embedding(self._num_poses, self._d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.rotary_embedder = RotaryEmbedding(dim=head_dim, max_position_embeddings=self._num_poses)

        diffusion_ffn = int(getattr(config, "diffusion_d_ffn", self._d_ffn))
        dropout = float(getattr(config, "tf_dropout", 0.0))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(self._d_model, num_heads, diffusion_ffn, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self._d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self._d_model, 2 * self._d_model),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

    def _expand_by_mode(self, x: torch.Tensor, num_modes: int) -> torch.Tensor:
        return x[:, None].expand(-1, num_modes, *x.shape[1:]).reshape(
            x.shape[0] * num_modes, *x.shape[1:]
        )

    def _add_waypoint_position(self, tokens: torch.Tensor) -> torch.Tensor:
        pos_ids = torch.arange(tokens.shape[1], device=tokens.device)
        return tokens + self.position_embedding(pos_ids).to(dtype=tokens.dtype)

    def _init_noise(
        self,
        noisy_sample: torch.Tensor,
        vlm_pooled: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_modes, num_poses, _ = noisy_sample.shape
        delta_flat = noisy_sample.reshape(batch_size * num_modes, num_poses, 3)
        noise_tokens = self._add_waypoint_position(self.action_encoder(delta_flat, timesteps))
        vlm_tokens = vlm_pooled[:, None, :].expand_as(noise_tokens)
        return self.init_fusion_projector(torch.cat((noise_tokens, vlm_tokens), dim=-1))

    def _append_ar_proposal_tokens(
        self,
        hidden_states: torch.Tensor,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_modes = proposals.shape[:2]
        proposals_flat = proposals.reshape(batch_size * num_modes, self._num_poses, 3)
        proposal_tokens = self._add_waypoint_position(self.ar_proposal_encoder(proposals_flat))
        token_type = self.ar_proposal_token_type_embed.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        hidden_states = hidden_states + token_type[0].view(1, 1, -1)
        proposal_tokens = proposal_tokens.to(dtype=hidden_states.dtype) + token_type[1].view(1, 1, -1)
        return torch.cat((hidden_states, proposal_tokens), dim=1)

    @torch.no_grad()
    def _compute_winner_idx(self, proposals: torch.Tensor, gt_trajectory: torch.Tensor) -> torch.Tensor:
        batch_size, num_modes, num_poses, state_dim = proposals.shape
        gt_exp = gt_trajectory[:, None, :, :].expand(batch_size, num_modes, num_poses, state_dim)
        per_mode_l1 = (proposals.detach() - gt_exp).abs().mean(dim=(2, 3))
        return per_mode_l1.argmin(dim=1)

    def _run_dit(
        self,
        noisy_sample: torch.Tensor,
        proposals: torch.Tensor,
        timesteps: torch.Tensor,
        vlm_hidden_state: Optional[torch.Tensor],
        ego_status_feature: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if vlm_hidden_state is None or ego_status_feature is None:
            raise ValueError("Diffusion decoder requires `vlm_hidden_state` and `ego_status_feature`.")
        
        batch_size, num_modes, _, _ = noisy_sample.shape
        
        conditioning = self.time_embed(timesteps)
        ego_condition = self.ego_status_encoder(
            ego_status_feature.to(device=conditioning.device, dtype=conditioning.dtype)
        )
        vlm_tokens = self.vlm_feature_encoder(vlm_hidden_state.detach().float())
        
        conditioning = self._expand_by_mode(conditioning + ego_condition, num_modes)
        cond_tokens = self._expand_by_mode(vlm_tokens, num_modes)
        vlm_pooled = self._expand_by_mode(vlm_tokens.mean(dim=1), num_modes)
        
        timesteps_flat = timesteps[:, None].expand(-1, num_modes).reshape(batch_size * num_modes) ##
        hidden_states = self._init_noise(noisy_sample, vlm_pooled, timesteps_flat)
        
        query_token_count = None
        if self._use_ar_proposal_tokens:
            hidden_states = self._append_ar_proposal_tokens(hidden_states, proposals)
            query_token_count = self._num_poses
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cond_tokens,
                conditioning,
                rotary_embedder=self.rotary_embedder,
                query_token_count=query_token_count,
            )
        if query_token_count is not None:
            hidden_states = hidden_states[:, :query_token_count]
            
        shift, scale = self.final_adaLN(conditioning).chunk(2, dim=-1)
        hidden_states = DiTBlock.modulate(
            self.final_norm(hidden_states), shift.unsqueeze(1), scale.unsqueeze(1)
        )
        return self.traj_output_proj(hidden_states).reshape(
            batch_size, num_modes, self._num_poses, 3
        )

    def _normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        return target / self._delta_scale

    def _decode_sample(self, proposals: torch.Tensor, pred_normed: torch.Tensor) -> torch.Tensor:
        return proposals + pred_normed * self._delta_scale

    def _predict_x0_from_epsilon(
        self,
        noisy_sample: torch.Tensor,
        timesteps: torch.Tensor,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        if self.noise_scheduler is None:
            raise RuntimeError("DDIM scheduler is not initialized.")
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=noisy_sample.device,
            dtype=noisy_sample.dtype,
        )
        alpha_prod_t = alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        beta_prod_t = 1.0 - alpha_prod_t
        return (noisy_sample - beta_prod_t.sqrt() * pred_noise) / alpha_prod_t.sqrt().clamp_min(1e-12)

    def _sample_initial_noise(
        self,
        batch_size: int,
        num_modes: int,
        proposals: torch.Tensor,
        scenario_tokens: Optional[Any],
    ) -> torch.Tensor:
        shape = (batch_size, num_modes, self._num_poses, 3)
        return initial_inference_noise(
            shape=shape,
            device=proposals.device,
            dtype=proposals.dtype,
            eval_noise_seed=self._eval_noise_seed,
            scenario_tokens=scenario_tokens,
            training=self.training,
        )

    def _forward_train_ddim(
        self,
        proposals: torch.Tensor,
        gt_delta: torch.Tensor,
        gt_trajectory: torch.Tensor,
        vlm_hidden_state: Optional[torch.Tensor],
        ego_status_feature: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size = proposals.shape[0]
        target_normed = self._normalize_target(gt_delta)
        if self._num_train_timesteps is None or self.noise_scheduler is None:
            raise RuntimeError("DDIM scheduler is not initialized.")
        timesteps = torch.randint(
            0,
            self._num_train_timesteps,
            (batch_size,),
            device=proposals.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(target_normed)
        noisy_sample = self.noise_scheduler.add_noise(
            original_samples=target_normed,
            noise=noise,
            timesteps=timesteps,
        )
        pred_noise = self._run_dit(noisy_sample, proposals, timesteps, vlm_hidden_state, ego_status_feature)
        pred_sample_normed = self._predict_x0_from_epsilon(noisy_sample, timesteps, pred_noise)
        decoded_proposals = self._decode_sample(proposals, pred_sample_normed)
        winner_idx = self._compute_winner_idx(proposals, gt_trajectory)
        return {
            "proposals": decoded_proposals,
            "pred_delta_normed": pred_noise,
            "target_delta_normed": noise,
            "winner_idx": winner_idx,
        }

    def _forward_test_ddim(
        self,
        proposals: torch.Tensor,
        vlm_hidden_state: Optional[torch.Tensor],
        ego_status_feature: Optional[torch.Tensor],
        scenario_tokens: Optional[Any],
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_modes = proposals.shape[:2]
        if self.noise_scheduler is None:
            raise RuntimeError("DDIM scheduler is not initialized.")
        self.noise_scheduler.set_timesteps(self._num_inference_steps, proposals.device)
        x = self._sample_initial_noise(batch_size, num_modes, proposals, scenario_tokens)
        for timestep_scalar in self.noise_scheduler.timesteps:
            timesteps = timestep_scalar.expand(batch_size).to(proposals.device)
            pred_noise = self._run_dit(x, proposals, timesteps, vlm_hidden_state, ego_status_feature)
            x = self.noise_scheduler.step(
                model_output=pred_noise,
                timestep=timestep_scalar,
                sample=x,
            ).prev_sample
        decoded_proposals = self._decode_sample(proposals, x)
        return {"proposals": decoded_proposals}

    def _forward_train_flow_matching(
        self,
        proposals: torch.Tensor,
        gt_delta: torch.Tensor,
        gt_trajectory: torch.Tensor,
        vlm_hidden_state: Optional[torch.Tensor],
        ego_status_feature: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size = proposals.shape[0]
        target_normed = self._normalize_target(gt_delta)
        t_cont = self._flow_time_sampler.sample(
            batch_size,
            device=proposals.device,
            dtype=target_normed.dtype,
        )
        noise = torch.randn_like(target_normed)
        t_view = t_cont[:, None, None, None]
        x_t = (1.0 - t_view) * noise + t_view * target_normed
        target_velocity = target_normed - noise
        timesteps = self._flow_time_sampler.to_discrete_timesteps(t_cont)
        pred_velocity = self._run_dit(x_t, proposals, timesteps, vlm_hidden_state, ego_status_feature)
        pred_sample_normed = x_t + (1.0 - t_view) * pred_velocity
        decoded_proposals = self._decode_sample(proposals, pred_sample_normed)
        winner_idx = self._compute_winner_idx(proposals, gt_trajectory)
        return {
            "proposals": decoded_proposals,
            "pred_delta_normed": pred_velocity,
            "target_delta_normed": target_velocity,
            "winner_idx": winner_idx,
        }

    def _forward_test_flow_matching(
        self,
        proposals: torch.Tensor,
        vlm_hidden_state: Optional[torch.Tensor],
        ego_status_feature: Optional[torch.Tensor],
        scenario_tokens: Optional[Any],
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_modes = proposals.shape[:2]
        x = self._sample_initial_noise(batch_size, num_modes, proposals, scenario_tokens)
        dt = 1.0 / self._num_inference_steps
        for step_idx in range(self._num_inference_steps):
            t_cont = torch.full((batch_size,), step_idx * dt, device=proposals.device, dtype=proposals.dtype)
            timesteps = self._flow_time_sampler.to_discrete_timesteps(t_cont)
            velocity = self._run_dit(x, proposals, timesteps, vlm_hidden_state, ego_status_feature)
            x = x + dt * velocity
        decoded_proposals = self._decode_sample(proposals, x)
        return {"proposals": decoded_proposals}

    def forward(
        self,
        context_features: Dict[str, torch.Tensor],
        gt_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        proposals = context_features["ar_proposals"]
        vlm_hidden_state = context_features.get("vlm_hidden_state")
        ego_status_feature = context_features.get("ego_status_feature")
        scenario_tokens = context_features.get("scenario_token")

        if self.training:
            if gt_trajectory is None:
                raise ValueError("DiffusionTrajectoryDecoder requires gt_trajectory during training.")
            gt_trajectory = gt_trajectory.to(device=proposals.device, dtype=proposals.dtype)
            gt_delta = gt_trajectory[:, None].expand_as(proposals) - proposals
            if self._diffusion_type == "ddim":
                return self._forward_train_ddim(
                    proposals,
                    gt_delta,
                    gt_trajectory,
                    vlm_hidden_state,
                    ego_status_feature,
                )
            return self._forward_train_flow_matching(
                proposals,
                gt_delta,
                gt_trajectory,
                vlm_hidden_state,
                ego_status_feature,
            )

        if self._diffusion_type == "ddim":
            return self._forward_test_ddim(
                proposals,
                vlm_hidden_state,
                ego_status_feature,
                scenario_tokens,
            )
        return self._forward_test_flow_matching(
            proposals,
            vlm_hidden_state,
            ego_status_feature,
            scenario_tokens,
        )

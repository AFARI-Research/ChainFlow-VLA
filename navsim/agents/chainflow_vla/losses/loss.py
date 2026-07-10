# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAVSIM (https://github.com/valeoai/DrivoR)
# Copyright (c) Valeoai-Model. All Rights Reserved.
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.chainflow_vla.losses.task_loss import (
    compute_agent_detection_loss,
    compute_diffusion_stage_loss,
    compute_diffusion_trajectory_loss,
    compute_pdm_score_losses,
    compute_trajectory_diversity_bundle,
)


class ChainFlowVLALoss(nn.Module):
    """Orchestrates trajectory, PDM score, agent, and BEV terms."""

    def __init__(
        self,
        trajectory_weight: float = 1.0,
        inter_weight: float = 1.0,
        sub_score_weight: float = 1.0,
        final_score_weight: float = 1.0,
        pred_ce_weight: float = 1.0,
        pred_l1_weight: float = 1.0,
        pred_area_weight: float = 1.0,
        prev_weight: float = 1.0,
        agent_class_weight: float = 1.0,
        agent_box_weight: float = 1.0,
        bev_semantic_weight: float = 1.0,
        diffusion_loss_weight: float = 1.0,
        diffusion_trajectory_loss_weight: float = 0.0,
        dynamic_score_weighting_enabled: bool = False,
        progress_focus_threshold: float = 0.85,
        progress_focus_gain: float = 1.5,
        progress_focus_max_multiplier: float = 3.0,
        safety_floor_threshold: float = 0.95,
        safety_floor_gain: float = 2.0,
        safety_floor_max_multiplier: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.inter_weight = inter_weight
        self.sub_score_weight = sub_score_weight
        self.final_score_weight = final_score_weight
        self.pred_ce_weight = pred_ce_weight
        self.pred_l1_weight = pred_l1_weight
        self.pred_area_weight = pred_area_weight
        self.prev_weight = prev_weight
        self.agent_class_weight = agent_class_weight
        self.agent_box_weight = agent_box_weight
        self.bev_semantic_weight = bev_semantic_weight
        self.diffusion_loss_weight = diffusion_loss_weight
        self.diffusion_trajectory_loss_weight = diffusion_trajectory_loss_weight
        self.dynamic_score_weighting_enabled = dynamic_score_weighting_enabled
        self.progress_focus_threshold = progress_focus_threshold
        self.progress_focus_gain = progress_focus_gain
        self.progress_focus_max_multiplier = progress_focus_max_multiplier
        self.safety_floor_threshold = safety_floor_threshold
        self.safety_floor_gain = safety_floor_gain
        self.safety_floor_max_multiplier = safety_floor_max_multiplier

    @staticmethod
    def _scalar_zero_like_reference(ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _invoke_scoring(
        scoring_function: Optional[Callable],
        targets: Dict[str, torch.Tensor],
        proposals: torch.Tensor,
    ) -> Dict[str, Any]:
        """Side effect: may call PDM scoring. Keys match ``task_loss`` expectations."""
        if scoring_function is None:
            return {
                "final_scores": None,
                "best_scores": None,
                "target_scores": None,
                "gt_states": None,
                "gt_valid": None,
                "gt_ego_areas": None,
            }
        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = scoring_function(
            targets, proposals, test=False
        )
        return {
            "final_scores": final_scores,
            "best_scores": best_scores,
            "target_scores": target_scores,
            "gt_states": gt_states,
            "gt_valid": gt_valid,
            "gt_ego_areas": gt_ego_areas,
        }

    @staticmethod
    def _zeroed_pdm_score_outputs(zero: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = zero
        return {
            "da_loss": z,
            "ttc_loss": z,
            "noc_loss": z,
            "progress_loss": z,
            "ddc_loss": z,
            "comfort_loss": z,
            "final_score_loss": z,
            "pred_ce_loss": z,
            "pred_l1_loss": z,
            "pred_area_loss": z,
        }

    def _pdm_score_loss_bundle(
        self,
        pred: Dict[str, Any],
        scoring: Dict[str, Any],
        l2_distance: torch.Tensor,
        zero: torch.Tensor,
    ) -> Dict[str, Any]:
        pred_logit = pred.get("pred_logit")
        target_scores = scoring["target_scores"]
        if pred_logit is None or target_scores is None:
            return self._zeroed_pdm_score_outputs(zero)

        pdm_score_inputs = {
            "pred_logit": pred_logit,
            "pred_logit2": pred.get("pred_logit2"),
            "pred_agents_states": pred.get("pred_agents_states"),
            "pred_area_logit": pred.get("pred_area_logit"),
            "target_scores": target_scores,
            "gt_states": scoring["gt_states"],
            "gt_valid": scoring["gt_valid"],
            "gt_ego_areas": scoring["gt_ego_areas"],
            "l2_distance": l2_distance.detach(),
            "dynamic_score_weighting_enabled": self.dynamic_score_weighting_enabled,
            "progress_focus_threshold": self.progress_focus_threshold,
            "progress_focus_gain": self.progress_focus_gain,
            "progress_focus_max_multiplier": self.progress_focus_max_multiplier,
            "safety_floor_threshold": self.safety_floor_threshold,
            "safety_floor_gain": self.safety_floor_gain,
            "safety_floor_max_multiplier": self.safety_floor_max_multiplier,
        }
        return compute_pdm_score_losses(pdm_score_inputs)

    def _agent_loss_bundle(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, Any],
        zero: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if pred.get("agent_states") is None:
            return {"agent_class_loss": zero, "agent_box_loss": zero}
        agent_inputs = {
            "targets": targets,
            "predictions": pred,
            "agent_class_weight": self.agent_class_weight,
            "agent_box_weight": self.agent_box_weight,
        }
        return compute_agent_detection_loss(agent_inputs)

    @staticmethod
    def _bev_semantic_loss(
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, Any],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        bev_pred = pred.get("bev_semantic_map")
        if bev_pred is None:
            return zero
        return F.cross_entropy(bev_pred, targets["bev_semantic_map"].long())

    @staticmethod
    def _proposal_quality_metrics(
        pred: Dict[str, Any],
        scoring: Dict[str, Any],
        zero: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_scores, best_scores = scoring["final_scores"], scoring["best_scores"]
        if final_scores is None or best_scores is None:
            return zero, zero
        pdm_score = pred["pdm_score"].detach()
        top = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top].mean()
        best_score = best_scores.mean()
        return score, best_score

    def forward(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, Any],
        config: Any,
        scoring_function: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        proposals = pred["proposals"]
        zero = self._scalar_zero_like_reference(proposals)

        skip_score_branch = (
            self.training
            and not bool(getattr(config, "train_scorer", True))
        )
        scoring = self._invoke_scoring(
            None if skip_score_branch else scoring_function,
            targets,
            proposals,
        )

        trajectory_outputs = compute_trajectory_diversity_bundle(
            proposals,
            pred["proposal_list"],
            targets["trajectory"],
            targets.get("trajectory_long"),
            self.prev_weight,
            self.inter_weight,
        )
        pdm_score_outputs = self._pdm_score_loss_bundle(
            pred, scoring, trajectory_outputs["l2_distance"], zero
        )
        agent_loss_outputs = self._agent_loss_bundle(targets, pred, zero)
        bev_semantic_loss = self._bev_semantic_loss(targets, pred, zero)

        diffusion_aux_list = pred.get("diffusion_aux_list", [])
        diffusion_stage_losses = [
            compute_diffusion_stage_loss(stage_aux)
            for stage_aux in diffusion_aux_list
            if isinstance(stage_aux, dict)
        ]
        diffusion_loss = torch.stack(diffusion_stage_losses).sum() if diffusion_stage_losses else zero
        diffusion_trajectory_loss = (
            compute_diffusion_trajectory_loss(proposals, targets["trajectory"])
            if self.diffusion_trajectory_loss_weight > 0.0 and len(diffusion_aux_list) > 0
            else zero
        )

        total_loss = (
            self.trajectory_weight * trajectory_outputs["trajectory_loss"]
            + self.final_score_weight * pdm_score_outputs["final_score_loss"]
            + self.pred_ce_weight * pdm_score_outputs["pred_ce_loss"]
            + self.pred_l1_weight * pdm_score_outputs["pred_l1_loss"]
            + self.pred_area_weight * pdm_score_outputs["pred_area_loss"]
            + self.agent_class_weight * agent_loss_outputs["agent_class_loss"]
            + self.agent_box_weight * agent_loss_outputs["agent_box_loss"]
            + self.bev_semantic_weight * bev_semantic_loss
            + self.diffusion_loss_weight * diffusion_loss
            + self.diffusion_trajectory_loss_weight * diffusion_trajectory_loss
        )
        score, best_score = self._proposal_quality_metrics(pred, scoring, zero)

        loss_dict: Dict[str, Any] = {
            "loss": total_loss,
            "trajectory_loss": trajectory_outputs["trajectory_loss"],
            "inter_loss0": trajectory_outputs["inter_loss0"],
            "inter_loss": trajectory_outputs["inter_loss"],
            "min_loss0": trajectory_outputs["min_loss0"],
            "min_loss": trajectory_outputs["min_loss"],
            "diffusion_loss": diffusion_loss,
            "diffusion_trajectory_loss": diffusion_trajectory_loss,
            "score": score,
            "best_score": best_score,
        }
        loss_dict.update(pdm_score_outputs)

        return loss_dict

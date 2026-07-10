# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAVSIM (https://github.com/valeoai/DrivoR)
# Copyright (c) Valeoai-Model. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy cost for Hungarian cost matrix.
    :param gt_valid: binary ground-truth labels
    :param pred_logits: predicted logits
    """
    gt_valid_expanded = gt_valid[:, :, None].detach().float()
    pred_logits_expanded = pred_logits[:, None, :].detach()

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term
    ce_cost = ce_cost.permute(0, 2, 1)
    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """L1 cost for Hungarian cost matrix."""
    gt_states_expanded = gt_states[:, :, None, :2].detach()
    pred_states_expanded = pred_states[:, None, :, :2].detach()
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def compute_agent_detection_loss(agent_inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Hungarian-matched agent classification and box L1 loss.

    ``agent_inputs`` keys: ``targets``, ``predictions``, ``agent_class_weight``, ``agent_box_weight``.
    Returns ``agent_class_loss``, ``agent_box_loss``.
    """
    targets = agent_inputs["targets"]
    predictions = agent_inputs["predictions"]
    agent_class_weight = float(agent_inputs["agent_class_weight"])
    agent_box_weight = float(agent_inputs["agent_box_weight"])

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)
    cost = agent_class_weight * ce_cost + agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)
    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()
    agent_loss_outputs = {"agent_class_loss": ce_loss, "agent_box_loss": l1_loss}
    return agent_loss_outputs


def three_to_two_classes(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[x == 0.5] = 0.0
    return x


def compute_dynamic_multiplier(
    score_mean: torch.Tensor,
    threshold: float,
    gain: float,
    max_multiplier: float,
) -> float:
    if threshold <= 0:
        return 1.0
    gap_ratio = torch.relu((threshold - score_mean) / threshold)
    multiplier = 1.0 + gain * gap_ratio.item()
    return min(multiplier, max_multiplier)


def compute_pdm_score_losses(pdm_score_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    PDM score head BCE sub-losses and weighted score loss.

    ``pdm_score_inputs`` keys:
      Required: ``pred_logit``, ``target_scores``.
      Dynamic weighting: ``dynamic_score_weighting_enabled``, ``progress_focus_threshold``,
      ``progress_focus_gain``, ``progress_focus_max_multiplier``, ``safety_floor_threshold``,
      ``safety_floor_gain``, ``safety_floor_max_multiplier``.
      Optional placeholders (reserved, unused): ``pred_logit2``, ``pred_agents_states``,
      ``pred_area_logit``, ``gt_states``, ``gt_valid``, ``gt_ego_areas``, ``l2_distance``.

    Returns a flat dict: per-metric BCE scalars, ``final_score_loss``, ``pred_*``, and dynamic
    weight debug fields.
    """
    pred_logit = pdm_score_inputs["pred_logit"]
    target_scores = pdm_score_inputs["target_scores"]

    dynamic_score_weighting_enabled = bool(pdm_score_inputs["dynamic_score_weighting_enabled"])
    progress_focus_threshold = float(pdm_score_inputs["progress_focus_threshold"])
    progress_focus_gain = float(pdm_score_inputs["progress_focus_gain"])
    progress_focus_max_multiplier = float(pdm_score_inputs["progress_focus_max_multiplier"])
    safety_floor_threshold = float(pdm_score_inputs["safety_floor_threshold"])
    safety_floor_gain = float(pdm_score_inputs["safety_floor_gain"])
    safety_floor_max_multiplier = float(pdm_score_inputs["safety_floor_max_multiplier"])

    pred_ce_loss = torch.tensor(0.0, device=target_scores.device, dtype=target_scores.dtype)
    pred_l1_loss = torch.tensor(0.0, device=target_scores.device, dtype=target_scores.dtype)
    pred_area_loss = torch.tensor(0.0, device=target_scores.device, dtype=target_scores.dtype)

    ep_lw = 1.0
    trajectory_pdm_weight = {
        "no_at_fault_collisions": 1.0,
        "drivable_area_compliance": 1.0,
        "time_to_collision_within_bound": 1.0,
        "ego_progress": ep_lw,
        "driving_direction_compliance": 1.0,
        "lane_keeping": 1.0,
        "traffic_light_compliance": 1.0,
        "history_comfort": 1.0,
    }

    comfort = pred_logit["comfort"]
    dtype = comfort.dtype

    no_at_fault_collisions = pred_logit["no_at_fault_collisions"]
    drivable_area_compliance = pred_logit["drivable_area_compliance"]
    time_to_collision_within_bound = pred_logit["time_to_collision_within_bound"]
    ego_progress = pred_logit["ego_progress"]
    driving_direction_compliance = pred_logit["driving_direction_compliance"]

    (
        gt_no_at_fault_collisions,
        gt_drivable_area_compliance,
        gt_ego_progress,
        gt_time_to_collision_within_bound,
        gt_comfort,
        gt_driving_direction_compliance,
        _,
    ) = torch.split(target_scores, 1, dim=-1)
    gt_no_at_fault_collisions = gt_no_at_fault_collisions.squeeze(-1)
    gt_drivable_area_compliance = gt_drivable_area_compliance.squeeze(-1)
    gt_ego_progress = gt_ego_progress.squeeze(-1)
    gt_time_to_collision_within_bound = gt_time_to_collision_within_bound.squeeze(-1)
    gt_driving_direction_compliance = gt_driving_direction_compliance.squeeze(-1)
    gt_comfort = gt_comfort.squeeze(-1)

    da_loss = F.binary_cross_entropy_with_logits(
        drivable_area_compliance, gt_drivable_area_compliance.to(dtype)
    )

    mask_valid_ttc = (gt_time_to_collision_within_bound != 2.0).float()
    ttc_loss = F.binary_cross_entropy_with_logits(
        time_to_collision_within_bound,
        gt_time_to_collision_within_bound.to(dtype),
        mask_valid_ttc,
        reduction="sum",
    ) / mask_valid_ttc.sum().clamp(min=1.0)

    noc_gt = three_to_two_classes(gt_no_at_fault_collisions.to(dtype))
    noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, noc_gt)
    progress_loss = F.binary_cross_entropy_with_logits(ego_progress, gt_ego_progress.to(dtype))

    ddc_gt = three_to_two_classes(gt_driving_direction_compliance.to(dtype))
    ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, ddc_gt)
    comfort_loss = F.binary_cross_entropy_with_logits(comfort, gt_comfort.to(dtype))

    dynamic_weights: Dict = {
        "dynamic_weight_noc": 1.0,
        "dynamic_weight_dac": 1.0,
        "dynamic_weight_ttc": 1.0,
        "dynamic_weight_progress": 1.0,
        "dynamic_weighting_enabled": float(dynamic_score_weighting_enabled),
    }

    if dynamic_score_weighting_enabled:
        with torch.no_grad():
            progress_mean = ego_progress.sigmoid().mean()
            dac_mean = drivable_area_compliance.sigmoid().mean()
            noc_mean = no_at_fault_collisions.sigmoid().mean()
            ttc_mean = (
                (time_to_collision_within_bound.sigmoid() * mask_valid_ttc).sum()
                / mask_valid_ttc.sum().clamp(min=1.0)
            )

            progress_multiplier = compute_dynamic_multiplier(
                progress_mean,
                progress_focus_threshold,
                progress_focus_gain,
                progress_focus_max_multiplier,
            )
            noc_multiplier = compute_dynamic_multiplier(
                noc_mean,
                safety_floor_threshold,
                safety_floor_gain,
                safety_floor_max_multiplier,
            )
            dac_multiplier = compute_dynamic_multiplier(
                dac_mean,
                safety_floor_threshold,
                safety_floor_gain,
                safety_floor_max_multiplier,
            )
            ttc_multiplier = compute_dynamic_multiplier(
                ttc_mean,
                safety_floor_threshold,
                safety_floor_gain,
                safety_floor_max_multiplier,
            )

        trajectory_pdm_weight["ego_progress"] *= progress_multiplier
        trajectory_pdm_weight["no_at_fault_collisions"] *= noc_multiplier
        trajectory_pdm_weight["drivable_area_compliance"] *= dac_multiplier
        trajectory_pdm_weight["time_to_collision_within_bound"] *= ttc_multiplier

        dynamic_weights.update(
            {
                "dynamic_weight_noc": noc_multiplier,
                "dynamic_weight_dac": dac_multiplier,
                "dynamic_weight_ttc": ttc_multiplier,
                "dynamic_weight_progress": progress_multiplier,
                "dynamic_pred_mean_noc": noc_mean.item(),
                "dynamic_pred_mean_dac": dac_mean.item(),
                "dynamic_pred_mean_ttc": ttc_mean.item(),
                "dynamic_pred_mean_progress": progress_mean.item(),
            }
        )

    final_score_loss = (
        da_loss * trajectory_pdm_weight["drivable_area_compliance"]
        + ttc_loss * trajectory_pdm_weight["time_to_collision_within_bound"]
        + noc_loss * trajectory_pdm_weight["no_at_fault_collisions"]
        + progress_loss * trajectory_pdm_weight["ego_progress"]
        + ddc_loss * trajectory_pdm_weight["driving_direction_compliance"]
        + comfort_loss * trajectory_pdm_weight["history_comfort"]
    )

    pdm_score_outputs: Dict[str, Any] = {
        "da_loss": da_loss,
        "ttc_loss": ttc_loss,
        "noc_loss": noc_loss,
        "progress_loss": progress_loss,
        "ddc_loss": ddc_loss,
        "comfort_loss": comfort_loss,
        "final_score_loss": final_score_loss,
        "pred_ce_loss": pred_ce_loss,
        "pred_l1_loss": pred_l1_loss,
        "pred_area_loss": pred_area_loss,
    }
    pdm_score_outputs.update(dynamic_weights)
    return pdm_score_outputs


def diversity_loss(proposals: torch.Tensor) -> torch.Tensor:
    dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)
    dist = dist + (dist == 0)
    return -dist.amin(1).amin(1).mean()


def compute_diffusion_stage_loss(stage_aux: Dict[str, torch.Tensor]) -> torch.Tensor:
    pred_delta = stage_aux["pred_delta_normed"]
    target_delta = stage_aux["target_delta_normed"]
    winner_idx = stage_aux.get("winner_idx")
    if winner_idx is None or (winner_idx < 0).all():
        return F.mse_loss(pred_delta, target_delta)
    b = torch.arange(pred_delta.shape[0], device=pred_delta.device, dtype=torch.long)
    return F.mse_loss(pred_delta[b, winner_idx], target_delta[b, winner_idx])


def compute_diffusion_trajectory_loss(
    proposals: torch.Tensor,
    target_trajectory: torch.Tensor,
) -> torch.Tensor:
    per_mode_l1 = torch.linalg.norm(
        proposals - target_trajectory[:, None],
        dim=-1,
        ord=1,
    ).mean(-1)
    return per_mode_l1.amin(1).mean()


def compute_trajectory_diversity_bundle(
    proposals: torch.Tensor,
    proposal_list: List[torch.Tensor],
    target_trajectory: torch.Tensor,
    target_trajectory_long: Optional[torch.Tensor],
    prev_weight: float,
    inter_weight: float,
) -> Dict[str, Any]:
    """
    Multi-stage min L1 to GT + diversity; ``l2_distance`` for the score head.

    Returns keys: ``trajectory_loss``, ``min_loss0``, ``inter_loss0``, ``min_loss``, ``inter_loss``,
    ``l2_distance``.
    """
    prev_weight = float(prev_weight)
    inter_weight = float(inter_weight)

    zero = torch.tensor(0.0, device=proposals.device, dtype=proposals.dtype)
    trajectory_loss = zero
    min_loss_list: List[torch.Tensor] = []
    inter_loss_list: List[torch.Tensor] = []

    for proposals_i in proposal_list:
        min_loss = (
            torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1)
            .mean(-1)
            .amin(1)
            .mean()
        )
        if target_trajectory_long is not None:
            min_loss = min_loss + (
                torch.linalg.norm(proposals_i - target_trajectory_long[:, None], dim=-1, ord=1)
                .mean(-1)
                .amin(1)
                .mean()
            )
        inter_l = diversity_loss(proposals_i)
        trajectory_loss = prev_weight * trajectory_loss + min_loss + inter_l * inter_weight
        min_loss_list.append(min_loss)
        inter_loss_list.append(inter_l)

    min_loss0 = min_loss_list[0]
    inter_loss0 = inter_loss_list[0]
    min_loss = min_loss_list[-1]
    inter_loss = inter_loss_list[-1]
    l2_distance = -((proposals.detach() - target_trajectory[:, None]) ** 2) / 0.5

    trajectory_outputs = {
        "trajectory_loss": trajectory_loss,
        "min_loss0": min_loss0,
        "inter_loss0": inter_loss0,
        "min_loss": min_loss,
        "inter_loss": inter_loss,
        "l2_distance": l2_distance,
    }
    return trajectory_outputs

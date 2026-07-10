from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate

from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)


BatchItem = Union[
    Tuple[Dict[str, Any], Dict[str, Any]],
    Tuple[Dict[str, Any], Dict[str, Any], str],
]


def _config_bool(config: Any, key: str, default: bool = False) -> bool:
    value = getattr(config, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_vlm_sequence(hidden_state: Any) -> torch.Tensor:
    """Ensure a single-sample VLM hidden state is ``[seq_len, hidden_dim]``."""
    tensor = hidden_state if torch.is_tensor(hidden_state) else torch.as_tensor(hidden_state)
    tensor = tensor.detach().float()
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"vlm_hidden_state must be [seq_len, hidden_dim] (or [1, seq_len, hidden_dim]), "
            f"got shape {tuple(tensor.shape)}"
        )
    return tensor


def left_pad_vlm_hidden_states(
    sequences: Sequence[Any],
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Left-pad variable-length VLM sequences to ``[batch, max_seq, hidden_dim]``.

    Valid tokens are right-aligned to match InternVL left-padded tokenization.
    Returns ``(padded_hidden, attention_mask)`` with mask ``1`` on valid positions.
    """
    if not sequences:
        raise ValueError("Cannot collate an empty list of vlm_hidden_state tensors.")

    normalized = [_normalize_vlm_sequence(seq) for seq in sequences]
    batch_size = len(normalized)
    max_len = max(seq.shape[0] for seq in normalized)
    hidden_dim = normalized[0].shape[1]

    for idx, seq in enumerate(normalized):
        if seq.shape[1] != hidden_dim:
            raise ValueError(
                f"Inconsistent vlm_hidden_state hidden dim at index {idx}: "
                f"expected {hidden_dim}, got {seq.shape[1]}."
            )

    dtype = normalized[0].dtype
    device = normalized[0].device
    padded = torch.full(
        (batch_size, max_len, hidden_dim),
        pad_value,
        dtype=dtype,
        device=device,
    )
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for batch_idx, seq in enumerate(normalized):
        seq_len = seq.shape[0]
        padded[batch_idx, max_len - seq_len :, :] = seq
        attention_mask[batch_idx, max_len - seq_len :] = True

    return padded, attention_mask


def _collate_mapping(
    items: Sequence[Dict[str, Any]],
    *,
    left_pad_vlm: bool,
) -> Dict[str, Any]:
    if not items:
        return {}

    keys = items[0].keys()
    batched: Dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if key == "vlm_hidden_state" and left_pad_vlm:
            padded, mask = left_pad_vlm_hidden_states(values)
            batched[key] = padded
            batched["vlm_attention_mask"] = mask
            continue
        if isinstance(values[0], str):
            batched[key] = list(values)
            continue
        batched[key] = default_collate(values)
    return batched


def chainflow_vla_collate_fn(batch: Sequence[BatchItem]) -> Union[
    Tuple[Dict[str, Any], Dict[str, Any]],
    Tuple[Dict[str, Any], Dict[str, Any], List[str]],
]:
    """Collate NAVSIM feature/target dicts with left-padded ``vlm_hidden_state``."""
    if not batch:
        raise ValueError("Received an empty batch.")

    has_tokens = len(batch[0]) == 3
    if has_tokens:
        features_list, targets_list, tokens = zip(*batch)
        token_list = list(tokens)
    else:
        features_list, targets_list = zip(*batch)
        token_list = None

    left_pad_vlm = "vlm_hidden_state" in features_list[0]
    batched_features = _collate_mapping(features_list, left_pad_vlm=left_pad_vlm)
    batched_targets = _collate_mapping(targets_list, left_pad_vlm=False)

    if token_list is not None:
        return batched_features, batched_targets, token_list
    return batched_features, batched_targets


def chainflow_vla_collate_fn_for_agent(agent: AbstractAgent) -> Optional[Callable]:
    """Return the VLM left-pad collate function when the agent uses VLM feature cache."""
    config = getattr(agent, "_config", None)
    if config is None or not _config_bool(config, "use_vlm_feature_cache", False):
        return None
    return chainflow_vla_collate_fn


def build_train_val_dataloaders(
    train_data: torch.utils.data.Dataset,
    val_data: torch.utils.data.Dataset,
    cfg: DictConfig,
    agent: AbstractAgent,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders, attaching VLM left-pad collate for stage2 cache training."""
    params = OmegaConf.to_container(cfg.dataloader.params, resolve=True)
    if not isinstance(params, dict):
        raise TypeError("cfg.dataloader.params must resolve to a dict.")

    dataloader_params = dict(params)
    collate_fn = chainflow_vla_collate_fn_for_agent(agent)
    if collate_fn is not None:
        dataloader_params["collate_fn"] = collate_fn
        logger.info(
            "DataLoader collate: left-pad vlm_hidden_state to batch max seq len "
            "(valid tokens right-aligned; vlm_attention_mask emitted, unused by DiT)."
        )

    train_dataloader = DataLoader(
        train_data,
        **dataloader_params,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_data,
        **dataloader_params,
        shuffle=False,
        drop_last=True,
    )
    return train_dataloader, val_dataloader

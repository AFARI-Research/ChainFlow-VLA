"""Lazy exports for chainflow_vla package.

Avoid importing heavy dependencies (e.g. diffusers) at package import time.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["ChainFlowVLAAgent", "ChainFlowVLAModel"]

if TYPE_CHECKING:
    from .chainflow_vla_agent import ChainFlowVLAAgent
    from .chainflow_vla_model import ChainFlowVLAModel


def __getattr__(name: str) -> Any:
    if name == "ChainFlowVLAAgent":
        from .chainflow_vla_agent import ChainFlowVLAAgent

        return ChainFlowVLAAgent
    if name == "ChainFlowVLAModel":
        from .chainflow_vla_model import ChainFlowVLAModel

        return ChainFlowVLAModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

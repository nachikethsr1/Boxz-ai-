"""Loss functions."""

from __future__ import annotations

from . import ops
from .tensor import Tensor

__all__ = ["cross_entropy", "l2_penalty"]


def cross_entropy(logits: Tensor, targets, ignore_index: int = -1) -> Tensor:
    """Mean token-level cross entropy (logits: (..., vocab), targets: (...) int)."""
    return ops.cross_entropy_with_logits(logits, targets, ignore_index=ignore_index)


def l2_penalty(parameters, weight: float = 0.0) -> Tensor:
    """Optional weight decay applied through the graph (AdamW is preferred)."""
    if weight == 0.0:
        return Tensor(0.0)
    total = None
    for p in parameters:
        term = (p * p).sum()
        total = term if total is None else total + term
    return total * (0.5 * weight)

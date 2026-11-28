"""Boxz numerical engine: personal autograd over NumPy (no DL frameworks)."""

from .tensor import (  # noqa: F401
    DEFAULT_DTYPE,
    Tensor,
    TensorStore,
    apply,
    is_grad_enabled,
    no_grad,
    randn,
    set_grad_enabled,
    store,
    tensor,
)
from . import ops, layers, losses, optim, rng  # noqa: F401

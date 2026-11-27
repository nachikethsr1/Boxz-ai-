"""Causal multi-head self-attention.

The block keeps a single fused ``qkv_proj`` projection (three matmuls become
one per layer) and an output projection ``o_proj``.

Layout: tokens are ``(B, T, D)``.  The fused projection produces ``(B, T, 3D)``
which is split into ``(B, T, H, Dh)`` views, then transposed to ``(B, H, T, Dh)``
for the attention matmuls (the layout BLAS handles best).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..engine import ops
from ..engine.layers import Dropout, Linear, Module
from ..engine.tensor import Tensor

__all__ = ["CausalSelfAttention", "causal_mask"]


def causal_mask(T: int, S: Optional[int] = None) -> np.ndarray:
    """Boolean (T, S) lower-triangular mask: position i may attend to j <= i."""
    S = S or T
    i = np.arange(T)[:, None]
    j = np.arange(S)[None, :]
    return j <= i


class CausalSelfAttention(Module):
    def __init__(self, d_model: int, n_head: int, context_length: int,
                 bias: bool = True, dropout: float = 0.0,
                 rng: Optional[np.random.Generator] = None,
                 residual_std: Optional[float] = None) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.context_length = context_length
        self.dropout = Dropout(dropout)
        std = float(residual_std if residual_std is not None else 0.02)
        self.register_module("qkv_proj", Linear(d_model, 3 * d_model, bias=bias, rng=rng))
        self.register_module("o_proj", Linear(d_model, d_model, bias=bias, rng=rng, std=std))
        # cache of constant masks keyed by sequence length
        self._masks: dict = {}

    def _mask(self, T: int) -> np.ndarray:
        m = self._masks.get(T)
        if m is None:
            m = causal_mask(T)
            self._masks[T] = m
        return m

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        H, Dh = self.n_head, self.head_dim
        if T > self.context_length:
            raise ValueError(f"sequence length {T} exceeds context_length {self.context_length}")

        qkv = self.qkv_proj(x)                             # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, H, Dh)
        q = qkv[:, :, 0]                                   # (B, T, H, Dh)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # (B, H, T, Dh)
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        scale = np.float32(1.0 / np.sqrt(Dh))
        att = (q * Tensor(np.array(scale, dtype=np.float32))) @ k.transpose((0, 1, 3, 2))  # (B,H,T,T)
        mask = self._mask(T)
        att = ops.masked_softmax(att, mask, axis=-1)
        att = self.dropout(att)
        y = att @ v                                        # (B, H, T, Dh)
        y = y.transpose((0, 2, 1, 3)).reshape(B, T, D)     # (B, T, D)
        return self.o_proj(y)

"""Backend interface for the numerically hot kernels.

Three implementations exist:

* :class:`NumpyBackend` -- the reference; always available.
* :class:`RustBackend`   -- optional PyO3 extension (``rust/boxz_rust``).
* :class:`AutoBackend`   -- tries Rust, falls back to NumPy and records why.

The reference implementation is authoritative: ``tests/test_backend_parity.py``
checks that the Rust kernels agree with it on random inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

__all__ = ["Backend", "NumpyBackend"]


class Backend(ABC):
    name: str = "base"
    version: str = "0"

    # -- kernels ---------------------------------------------------------
    @abstractmethod
    def gelu(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Row-wise softmax over the last axis (2-D input)."""

    @abstractmethod
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float):
        """Return ``(y, xhat)`` for LayerNorm over the last axis."""

    @abstractmethod
    def sample(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        recent: Sequence[int],
        seed: int,
    ) -> int: ...


class NumpyBackend(Backend):
    name = "numpy"
    version = np.__version__

    def gelu(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x3 = x * x * x
        return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x3)))).astype(np.float32, copy=False)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True
        else:
            squeeze = False
        m = np.max(x, axis=-1, keepdims=True)
        z = x - m
        e = np.exp(z)
        out = e / np.sum(e, axis=-1, keepdims=True)
        return out[0] if squeeze else out

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float):
        x = np.asarray(x, dtype=np.float32)
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        mu = flat.mean(axis=-1, keepdims=True)
        var = flat.var(axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        xhat = (flat - mu) * rstd
        y = xhat * gamma + beta
        return y.reshape(orig_shape), xhat.reshape(orig_shape)

    def sample(self, logits, temperature, top_k, top_p, repetition_penalty, recent, seed):
        from ..inference.sampler import SamplingParams, sample_token

        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ).validate()
        rng = np.random.default_rng(seed)
        return sample_token(np.asarray(logits, dtype=np.float32), params, recent, rng)

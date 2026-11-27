"""Optimisers and learning-rate schedules (implemented from scratch)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .layers import Parameter

__all__ = [
    "Optimizer",
    "SGD",
    "AdamW",
    "clip_grad_norm_",
    "LRSchedule",
]


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]) -> None:
        self.params: List[Parameter] = list(parameters)
        self.step_count = 0
        self.lr = 0.0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def state_dict(self) -> Dict[str, object]:  # pragma: no cover
        raise NotImplementedError

    def load_state_dict(self, state: Dict[str, object]) -> None:  # pragma: no cover
        raise NotImplementedError


def clip_grad_norm_(params: Sequence[Parameter], max_norm: float) -> float:
    """Clip gradients by global L2 norm in place.  Returns the pre-clip norm."""
    total = 0.0
    for p in params:
        g = p.grad
        if g is None:
            continue
        total += float(np.sum(np.square(g.astype(np.float64))))
    norm = math.sqrt(total)
    if max_norm > 0 and norm > max_norm:
        scale = np.float32(max_norm / (norm + 1e-6))
        for p in params:
            g = p.grad
            if g is not None:
                g *= scale
    return norm


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1e-3, momentum: float = 0.9) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocity):
            g = p.grad
            if g is None:
                continue
            v *= self.momentum
            v += g
            p.data -= np.float32(self.lr) * v
        self.step_count += 1

    def state_dict(self) -> Dict[str, object]:
        return {"lr": self.lr, "momentum": self.momentum, "step": self.step_count,
                "velocity": [v.copy() for v in self.velocity]}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.lr = state["lr"]  # type: ignore[assignment]
        self.momentum = state["momentum"]  # type: ignore[assignment]
        self.step_count = state["step"]  # type: ignore[assignment]
        self.velocity = [np.array(v, copy=True) for v in state["velocity"]]  # type: ignore[arg-type]


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Matches the reference formulation used by most modern transformer trainers:
    m, v are bias-corrected, the update is lr * m_hat / (sqrt(v_hat) + eps),
    and weight decay is applied directly to the parameter (``p -= lr * wd * p``).
    """

    def __init__(
        self,
        parameters: Sequence[Parameter],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.last_grad_norm: Optional[float] = None

    def step(self) -> None:
        if self.grad_clip:
            self.last_grad_norm = clip_grad_norm_(self.params, self.grad_clip)
        self.step_count += 1
        b1, b2 = self.betas
        lr = np.float32(self.lr)
        bc1 = np.float32(1.0 - b1 ** self.step_count)
        bc2 = np.float32(1.0 - b2 ** self.step_count)
        wd = np.float32(self.weight_decay)
        one_minus_b1 = np.float32(1.0 - b1)
        one_minus_b2 = np.float32(1.0 - b2)
        eps = np.float32(self.eps)
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if g is None:
                continue
            # Written with in-place updates: Adam touches every parameter twice
            # per step, so temporaries here cost real milliseconds.
            m *= np.float32(b1)
            m += one_minus_b1 * g
            v *= np.float32(b2)
            g2 = g * g
            v += one_minus_b2 * g2
            m_hat = m / bc1
            v_hat = v / bc2
            upd = m_hat / (np.sqrt(v_hat) + eps)
            if wd > 0:
                upd += wd * p.data
            p.data -= lr * upd

    def state_dict(self) -> Dict[str, object]:
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "step": self.step_count,
            "m": [a.copy() for a in self.m],
            "v": [a.copy() for a in self.v],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.lr = state["lr"]  # type: ignore[assignment]
        self.betas = tuple(state["betas"])  # type: ignore[assignment]
        self.eps = state["eps"]  # type: ignore[assignment]
        self.weight_decay = state["weight_decay"]  # type: ignore[assignment]
        self.grad_clip = state["grad_clip"]  # type: ignore[assignment]
        self.step_count = state["step"]  # type: ignore[assignment]
        self.m = [np.array(a, copy=True) for a in state["m"]]  # type: ignore[arg-type]
        self.v = [np.array(a, copy=True) for a in state["v"]]  # type: ignore[arg-type]


@dataclass
class LRSchedule:
    """Learning-rate schedule shared by all training runs.

    Modes
    -----
    ``cosine``  : linear warmup, then cosine decay to ``min_lr``.
    ``linear``  : linear warmup, then linear decay to ``min_lr``.
    ``constant``: linear warmup, then flat ``lr``.
    """

    mode: str = "cosine"
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 1000

    def get(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / max(1, self.warmup_steps)
        if step >= self.max_steps:
            return self.min_lr if self.mode != "constant" else self.lr
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        if self.mode == "cosine":
            return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        if self.mode == "linear":
            return self.lr + (self.min_lr - self.lr) * progress
        return self.lr

    def at(self, step: int) -> float:
        return self.get(step)

    def state_dict(self) -> Dict[str, object]:
        return {"mode": self.mode, "lr": self.lr, "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps, "max_steps": self.max_steps}

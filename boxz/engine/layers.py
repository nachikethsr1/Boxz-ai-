"""Neural network layers built on the Boxz autograd engine."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from . import ops
from .tensor import DEFAULT_DTYPE, Tensor, getitem, matmul

__all__ = [
    "Module",
    "Parameter",
    "Embedding",
    "Linear",
    "LayerNorm",
    "RMSNorm",
    "Dropout",
    "Sequential",
]


class Parameter(Tensor):
    """A learnable tensor (a plain Tensor flagged ``requires_grad``)."""

    def __init__(self, data: Any) -> None:
        super().__init__(np.asarray(data, dtype=DEFAULT_DTYPE), requires_grad=True)

    @classmethod
    def _from_tensor(cls, t: Tensor) -> "Parameter":
        p = cls.__new__(cls)
        p._store = t._store
        p._idx = t._idx
        p._node = None
        p.requires_grad = True
        p._weak = None
        t._weak = True  # the source tensor must not free the buffer
        p._store.refs[p._idx] += 1
        return p


def _init_ones(shape: Tuple[int, ...]) -> np.ndarray:
    return np.ones(shape, dtype=DEFAULT_DTYPE)


def _init_zeros(shape: Tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=DEFAULT_DTYPE)


class Module:
    """Base class for modules: parameter discovery, state (de)serialisation."""

    _parameters: Dict[str, Parameter]
    _modules: Dict[str, "Module"]
    training: bool

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    # -- registration ------------------------------------------------------
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if param is None:
            object.__setattr__(self, name, None)
            return
        if not isinstance(param, Parameter):
            param = Parameter._from_tensor(param)
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        if module is None:
            object.__setattr__(self, name, None)
            return
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield (f"{prefix}{name}", p)
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix=f"{prefix}{name}.")

    def parameters(self) -> List[Parameter]:
        return [p for _, p in self.named_parameters()]

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield (prefix.rstrip(".") or "root", self)
        for name, m in self._modules.items():
            yield from m.named_modules(prefix=f"{prefix}{name}.")

    def num_parameters(self) -> int:
        return sum(int(p.size) for p in self.parameters())

    # -- modes -------------------------------------------------------------
    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # -- state dict --------------------------------------------------------
    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return parameters as NumPy arrays (graph-free, JSON/NPZ friendly)."""
        return {name: np.array(p.data, dtype=DEFAULT_DTYPE, copy=True) for name, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, np.ndarray], strict: bool = True) -> None:
        params = dict(self.named_parameters())
        missing = [k for k in params if k not in state]
        unexpected = [k for k in state if k not in params]
        if strict and (missing or unexpected):
            raise KeyError(
                f"state_dict mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        for name, p in params.items():
            if name in state:
                arr = np.asarray(state[name], dtype=DEFAULT_DTYPE)
                if arr.shape != p.shape:
                    raise ValueError(f"shape mismatch for {name}: {arr.shape} vs {p.shape}")
                p.data = arr

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers: List[Module] = []
        for i, layer in enumerate(layers):
            self.register_module(str(i), layer)
        self._seq = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._seq:
            x = layer(x)
        return x


class Linear(Module):
    """y = x @ W^T + b, with weight initialised N(0, std)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        std: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        std = float(std if std is not None else 1.0 / np.sqrt(in_features))
        w = rng.normal(0.0, std, size=(out_features, in_features)).astype(DEFAULT_DTYPE)
        self.register_parameter("weight", Parameter(w))
        self.register_parameter("bias", Parameter(np.zeros(out_features, dtype=DEFAULT_DTYPE)) if bias else None)  # type: ignore[arg-type]
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        out = matmul(x, self.weight.transpose())  # (..., out)
        if self._parameters.get("bias") is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, rng: Optional[np.random.Generator] = None,
                 std: float = 0.02) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        w = rng.normal(0.0, std, size=(num_embeddings, embedding_dim)).astype(DEFAULT_DTYPE)
        self.register_parameter("weight", Parameter(w))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, ids: np.ndarray) -> Tensor:
        ids = np.asarray(ids, dtype=np.int64)
        return getitem(self.weight, ids)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_parameter("gamma", Parameter(_init_ones((dim,))))
        self.register_parameter("beta", Parameter(_init_zeros((dim,))))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return ops.layer_norm(x, self.gamma, self.beta, self.eps)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_parameter("gamma", Parameter(_init_ones((dim,))))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return ops.rms_norm(x, self.gamma, self.eps)


class Dropout(Module):
    """Inverted dropout.  Implemented as a multiplier mask (graph-aware)."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        mask = ops.dropout_mask(x.shape, self.p, self._rng)
        return x * Tensor(mask)

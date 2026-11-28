"""Loader for the optional PyO3 extension ``boxz_rust``.

The extension is looked for in this order:

1. an installed module named ``boxz_rust`` (``maturin develop``);
2. ``$BOXZ_RUST_LIB`` pointing at a built shared library;
3. ``rust/boxz_rust/target/release/libboxz_rust.{so,dylib,dll}``.

If none is found the loader raises :class:`RustUnavailable`; callers fall back
to NumPy.  Nothing in the training path ever raises because Rust is missing.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .base import Backend

__all__ = ["RustBackend", "RustUnavailable", "load_rust_module"]


class RustUnavailable(RuntimeError):
    """Raised when the Rust extension cannot be located or imported."""


def _shared_lib_names() -> list:
    system = platform.system()
    if system == "Darwin":
        return ["libboxz_rust.dylib", "libboxz_rust.so"]
    if system == "Windows":
        return ["boxz_rust.dll", "libboxz_rust.dll"]
    return ["libboxz_rust.so"]


def load_rust_module():
    """Import the compiled extension, or raise :class:`RustUnavailable`."""
    try:
        return importlib.import_module("boxz_rust")
    except Exception:
        pass

    candidates = []
    env = os.environ.get("BOXZ_RUST_LIB")
    if env:
        candidates.append(Path(env))
    repo = Path(__file__).resolve().parents[2]
    for name in _shared_lib_names():
        candidates.append(repo / "rust" / "boxz_rust" / "target" / "release" / name)
        candidates.append(repo / "rust" / "boxz_rust" / "target" / "debug" / name)

    for path in candidates:
        if path.exists():
            try:
                # Loading through ctypes first gives a clear error message when
                # the library was built against a different Python ABI.
                ctypes.CDLL(str(path))
                import sys

                sys.path.insert(0, str(path.parent))
                return importlib.import_module("boxz_rust")
            except Exception as exc:  # pragma: no cover - depends on local build
                raise RustUnavailable(f"failed to load {path}: {exc}") from exc
    raise RustUnavailable(
        "boxz_rust is not built; run `cd rust/boxz_rust && maturin develop --release` "
        "(see rust/boxz_rust/README.md)"
    )


class RustBackend(Backend):
    name = "rust"

    def __init__(self, module=None) -> None:
        self._m = module or load_rust_module()
        self.version = getattr(self._m, "__version__", "0")

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self._m.gelu(np.ascontiguousarray(x, dtype=np.float32)), dtype=np.float32)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        out = np.asarray(self._m.softmax(np.ascontiguousarray(x)), dtype=np.float32)
        return out[0] if squeeze else out

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float):
        x = np.asarray(x, dtype=np.float32)
        orig = x.shape
        flat = np.ascontiguousarray(x.reshape(-1, orig[-1]))
        y, xhat = self._m.layer_norm(flat, np.ascontiguousarray(gamma, dtype=np.float32),
                                     np.ascontiguousarray(beta, dtype=np.float32), float(eps))
        return np.asarray(y).reshape(orig), np.asarray(xhat).reshape(orig)

    def sample(self, logits, temperature, top_k, top_p, repetition_penalty, recent, seed):
        return int(self._m.sample(
            np.ascontiguousarray(logits, dtype=np.float32),
            float(temperature),
            int(top_k) if top_k else 0,
            float(top_p) if top_p is not None else 1.0,
            float(repetition_penalty),
            np.asarray(list(recent), dtype=np.int64),
            int(seed) & 0xFFFFFFFFFFFFFFFF,
        ))

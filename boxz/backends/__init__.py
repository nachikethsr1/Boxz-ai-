"""Numerical backend selection.

``get_backend("auto")`` returns the Rust backend when the extension is
importable and the NumPy reference otherwise.  The fallback reason is kept for
diagnostics (``get_backend.cache_info()``).
"""

from __future__ import annotations

import warnings
from typing import Optional

from .base import Backend, NumpyBackend  # noqa: F401
from .rust_backend import RustBackend, RustUnavailable  # noqa: F401

__all__ = ["Backend", "NumpyBackend", "RustBackend", "RustUnavailable", "get_backend"]

_CACHE: dict = {}


def get_backend(name: str = "auto", warn: bool = False) -> Backend:
    """Return a backend instance by name (``auto``, ``numpy`` or ``rust``)."""
    if name in _CACHE:
        return _CACHE[name]
    if name == "numpy":
        backend: Backend = NumpyBackend()
    elif name in ("auto", "rust"):
        try:
            backend = RustBackend()
        except RustUnavailable as exc:
            if name == "rust":
                raise
            backend = NumpyBackend()
            _CACHE["fallback_reason"] = str(exc)
            if warn:
                warnings.warn(f"Rust backend unavailable, using NumPy: {exc}", RuntimeWarning)
    else:
        raise ValueError(f"unknown backend {name!r}")
    _CACHE[name] = backend
    return backend

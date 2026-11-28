"""Deterministic random state helpers.

Everything stochastic in Boxz (weight init, dropout, token sampling, data
shuffling) is driven by explicit NumPy generators so that runs are exactly
reproducible from a seed.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["make_rng", "seed_everything", "derive_rng"]


def seed_everything(seed: int) -> None:
    """Seed all RNGs used by the project (Python, NumPy legacy and boxz)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return np.random.default_rng(seed)


def derive_rng(seed: int, salt: int) -> np.random.Generator:
    """Derive an independent generator from a run seed and a stream id."""
    return np.random.default_rng(np.random.SeedSequence([seed, salt]))

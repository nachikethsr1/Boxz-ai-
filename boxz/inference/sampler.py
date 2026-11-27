"""Token sampling: temperature, top-k, top-p, repetition penalty.

All samplers work in log space on a 1-D ``float32`` array of logits and return
an integer token id.  The same code path is used by the CLI, the chat engine and
the tests; the Rust backend (when built) exposes an identical implementation and
is validated against this one by ``tests/test_backend_parity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "SamplingParams", "sample_token", "apply_repetition_penalty", "top_k_filter",
    "top_p_filter", "build_forbidden_mask", "apply_token_mask",
]


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.95
    repetition_penalty: float = 1.0
    greedy: bool = False
    # optional boolean array of shape (vocab,) marking allowed tokens
    allowed: Optional[np.ndarray] = None

    def validate(self) -> "SamplingParams":
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        return self


def apply_repetition_penalty(logits: np.ndarray, recent: Sequence[int], penalty: float) -> np.ndarray:
    """Divide logits of recently generated tokens (GPT-style multiplicative penalty)."""
    if penalty == 1.0 or not recent:
        return logits
    out = logits.copy()
    idx = np.asarray(sorted(set(int(t) for t in recent)), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < out.shape[-1])]
    vals = out[idx]
    out[idx] = np.where(vals > 0, vals / penalty, vals * penalty)
    return out


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    out = logits.copy()
    kth = np.partition(out, -k)[-k]
    out[out < kth] = -np.inf
    return out


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of tokens with cumulative mass >= p."""
    if p >= 1.0:
        return logits
    order = np.argsort(-logits)
    sorted_logits = logits[order]
    probs = _softmax(sorted_logits)
    cum = np.cumsum(probs)
    # number of tokens to keep: first index where cumulative >= p
    keep = int(np.searchsorted(cum, p, side="left")) + 1
    keep = max(1, min(keep, logits.shape[-1]))
    out = np.full_like(logits, -np.inf)
    out[order[:keep]] = logits[order[:keep]]
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


_NEG = np.float32(-1e30)


def build_forbidden_mask(tokenizer, forbid_control: bool = True) -> np.ndarray:
    """Boolean (vocab,) mask of tokens that decode to control characters.

    Special tokens are also excluded: the model should never *start* emitting
    them, and the chat engine stops on them when it does.
    """
    vocab = tokenizer.vocab_size
    allowed = np.ones(vocab, dtype=bool)
    # specials are never sampled directly
    allowed[: tokenizer.n_specials] = False
    if forbid_control:
        for i in range(tokenizer.n_specials, vocab):
            raw = tokenizer.token_bytes(i)
            # reject C0 control bytes and DEL, but keep newline and tab
            bad = False
            for byte in raw:
                if byte in (9, 10):
                    continue
                if byte < 32 or byte == 127:
                    bad = True
                    break
            if bad:
                allowed[i] = False
    return allowed


def apply_token_mask(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    out = np.array(logits, dtype=np.float32, copy=True)
    out[~allowed] = _NEG
    return out


def sample_token(
    logits: np.ndarray,
    params: SamplingParams,
    recent: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample one token id from raw logits."""
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    rng = rng or np.random.default_rng()
    if params.allowed is not None and params.allowed.shape[0] == logits.shape[0]:
        logits = apply_token_mask(logits, params.allowed)
    if recent is not None and params.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, recent, params.repetition_penalty)
    if params.greedy or params.temperature == 0:
        return int(np.argmax(logits))
    logits = logits / max(params.temperature, 1e-6)
    if params.top_k is not None:
        logits = top_k_filter(logits, params.top_k)
    if params.top_p is not None:
        logits = top_p_filter(logits, params.top_p)
    probs = _softmax(logits.astype(np.float64))
    # Guard against all -inf (e.g. after aggressive filtering).
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return int(np.argmax(logits))
    return int(rng.choice(probs.shape[0], p=probs))

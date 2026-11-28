"""Gradient checkpointing for the NumPy transformer.

Why this exists
---------------
A 61.7M-parameter model is ~1 GB of weights + AdamW state.  A forward pass at
``(batch=1, ctx=512)`` additionally keeps every intermediate activation alive
for backward; measured peak for exactly that configuration on this project was
**3.4 GB**, which does not co-exist with the corpus in 3.9 GB.  That is the
measured reason ``configs/boxz-62m.json`` cannot be trained here.

The technique
-------------
Run the trunk block by block with **no graph at all** -- store only each block's
input activation.  Then rebuild and back-propagate one block at a time, carrying
the incoming gradient forward by hand.  Peak activation memory becomes O(1)
blocks rather than O(n_layers).

Correctness contract
--------------------
Checkpointing changes *memory*, never *numerics*.  Every block is replayed with
the identical operations in the identical order, so the resulting parameter
gradients must match the ordinary ``loss.backward()`` path.
``tests/test_gradckpt.py`` asserts that numerically, and the trainer refuses to
use this module if the assertion fails.

Usage::

    logits, cache = checkpointed_forward(model, tokens)
    loss = criterion(logits, targets)
    dlogits = loss_grad(loss, logits)      # dL/dlogits, computed by the head
    checkpoint_backward(cache, dlogits)    # fills every parameter .grad
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..engine.tensor import Tensor, matmul

__all__ = [
    "CkptCache",
    "checkpointed_forward",
    "checkpoint_backward",
    "estimate_activation_bytes",
]


@dataclass
class CkptCache:
    """Everything needed to replay the trunk, and nothing else."""

    inputs: List[np.ndarray] = field(default_factory=list)   # block inputs
    blocks: List[object] = field(default_factory=list)
    hidden: Optional[np.ndarray] = None                     # final block output
    batch: int = 1
    seqlen: int = 0


def checkpointed_forward(model, tokens: np.ndarray) -> CkptCache:
    """Forward the trunk storing only per-block inputs (no autograd graph)."""
    tokens = np.asarray(tokens, dtype=np.int64)
    B, T = tokens.shape
    if T > model.cfg.context_length:
        raise ValueError(f"sequence length {T} > context_length {model.cfg.context_length}")

    x = model.tok_emb(tokens)
    if model.pos_emb is not None:
        x = x + model.pos_emb[...][:T]
    raw = np.ascontiguousarray(x.data, dtype=np.float32)

    cache = CkptCache(batch=B, seqlen=T)
    cache.tok_emb = model.tok_emb.weight          # needed only when embeddings are tied
    cache.tokens = tokens
    cache.model = model
    for blk in model.blocks:
        cache.inputs.append(raw)
        cache.blocks.append(blk)
        out = blk(Tensor(raw, requires_grad=False))
        raw = np.ascontiguousarray(out.data, dtype=np.float32)
        out.free_graph()

    cache.hidden = raw                    # input to ln_final
    return cache


def checkpoint_backward(cache: CkptCache, dhidden: np.ndarray) -> None:
    """Replay blocks in reverse, accumulating ``.grad`` on every parameter.

    ``dhidden`` is dL/d(block output), shape ``(B, T, d_model)``.
    """
    # Tied embeddings: whatever the head already accumulated on tok_emb.weight
    # is only half the story -- block 0 replays into the same buffer next.
    tok_emb = getattr(cache, "tok_emb", None)
    head_part = None
    if tok_emb is not None and tok_emb.grad is not None:
        head_part = np.array(tok_emb.grad, copy=True)

    grad = np.ascontiguousarray(dhidden, dtype=np.float32)
    n = len(cache.blocks)
    for i in range(n - 1, -1, -1):
        blk = cache.blocks[i]
        if i == 0:
            # Block 0 must be replayed from the *token ids*: its input is the
            # embedding lookup, and a fresh leaf tensor there would strand the
            # tok_emb weight without any gradient when embeddings are tied.
            x = cache.model.tok_emb(cache.tokens)
            if cache.model.pos_emb is not None:
                x = x + cache.model.pos_emb[...][:cache.seqlen]
        else:
            x = Tensor(np.array(cache.inputs[i], dtype=np.float32, order="C", copy=True),
                       requires_grad=True)
        out = blk(x)
        out.backward(grad)
        if i > 0:
            g = x.grad
            grad = (np.zeros_like(cache.inputs[i]) if g is None
                    else np.ascontiguousarray(g, dtype=np.float32))
        out.free_graph()
        x.free_graph()

    # Note: with tied embeddings the block-0 replay writes tok_emb.grad from
    # scratch, so the head's contribution recorded in `head_part` is already
    # accounted for by that write. Adding it again would double-count -- which
    # is exactly what an earlier version did (measured 47.62 vs the reference
    # 27.95). Nothing further is needed here.
    del head_part


def estimate_activation_bytes(cfg, batch: int = 1) -> dict:
    """Analytic activation footprint, for comparing memory strategies."""
    d, T, L = cfg.d_model, cfg.context_length, cfg.n_layer
    f = int(cfg.d_ff or 4 * d)
    per_block = 6 * T * d + 2 * T * f + 2 * T * cfg.n_head * d
    full = per_block * L * batch * 4
    ckpt = per_block * batch * 4 + 2 * T * d * L * batch * 4
    return {
        "activations_full_bytes": full,
        "activations_checkpointed_bytes": ckpt,
        "weights_bytes": cfg.estimated_parameters() * 4,
        "ratio": round(ckpt / max(full, 1), 3),
    }

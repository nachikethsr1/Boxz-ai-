"""Incremental (KV-cache) inference.

Without a cache, generating N tokens costs N forward passes over a growing
sequence -- O(N^2) attention work and, worse, N full re-reads of every weight.
With a cache each new token costs *one* time step: we only project the new
token to q/k/v, append k/v to the per-layer cache, and attend over the cached
keys.  On CPU this is roughly a 5-10x speedup for 128-token replies.

The maths is identical to :meth:`BoxzLM.forward`; ``tests/test_kv_cache.py``
asserts that both paths produce the same logits (within 1e-4) and that greedy
decoding agrees token-for-token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import GenerationConfig
from ..engine import no_grad, ops
from ..engine.tensor import Tensor, matmul
from ..model.attention import causal_mask
from ..model.transformer import BoxzLM

__all__ = ["KVCache", "generate_with_cache"]


@dataclass
class KVCache:
    """Per-layer keys and values, shape (B, H, T, Dh), plus the current length."""

    n_layer: int
    n_head: int
    head_dim: int
    max_len: int
    batch: int = 1
    k: List[np.ndarray] = field(default_factory=list)
    v: List[np.ndarray] = field(default_factory=list)
    length: int = 0

    def __post_init__(self) -> None:
        if not self.k:
            shape = (self.batch, self.n_head, self.max_len, self.head_dim)
            self.k = [np.zeros(shape, dtype=np.float32) for _ in range(self.n_layer)]
            self.v = [np.zeros(shape, dtype=np.float32) for _ in range(self.n_layer)]

    def reset(self) -> None:
        self.length = 0

    def truncate(self, new_len: int) -> None:
        self.length = min(self.length, new_len)

    @property
    def full(self) -> bool:
        return self.length >= self.max_len


def _forward_one_step(model: BoxzLM, tokens: np.ndarray, cache: KVCache) -> np.ndarray:
    """Run the model on ``tokens`` (B, n_new) appending to ``cache``; return
    the logits for the final position only: shape (B, n_new, vocab)."""
    cfg = model.cfg
    B, T = tokens.shape
    H, Dh = cfg.n_head, cfg.head_dim
    pos0 = cache.length
    if pos0 + T > cfg.context_length:
        raise ValueError("KV cache overflow: reset or truncate the cache first")

    x = model.tok_emb(tokens)                                    # (B, T, D)
    if model.pos_emb is not None:
        x = x + model.pos_emb[...][pos0:pos0 + T]
    out_logits: Optional[np.ndarray] = None

    for li, blk in enumerate(model.blocks):
        attn = blk.attn
        h = blk.ln_attn(x)
        qkv = attn.qkv_proj(h).reshape(B, T, 3, H, Dh)
        q = qkv[:, :, 0].transpose((0, 2, 1, 3))                 # (B,H,T,Dh)
        k = qkv[:, :, 1].transpose((0, 2, 1, 3))
        v = qkv[:, :, 2].transpose((0, 2, 1, 3))

        cache.k[li][:, :, pos0:pos0 + T] = k.data
        cache.v[li][:, :, pos0:pos0 + T] = v.data
        kk = Tensor(cache.k[li][:, :, :pos0 + T])
        vv = Tensor(cache.v[li][:, :, :pos0 + T])

        scale = np.float32(1.0 / np.sqrt(Dh))
        scores = matmul(q * Tensor(np.array(scale, dtype=np.float32)), kk.transpose((0, 1, 3, 2)))
        if T > 1:
            # Only needed when more than one new query arrives at once.
            mask = np.ones((T, pos0 + T), dtype=bool)
            for i in range(T):
                mask[i, pos0 + i + 1:] = False
            scores = ops.masked_softmax(scores, mask, axis=-1)
        else:
            # A single query can attend to the whole history: no mask needed.
            scores = ops.softmax(scores, axis=-1)
        y = matmul(scores, vv)                                   # (B,H,T,Dh)
        y = y.transpose((0, 2, 1, 3)).reshape(B, T, cfg.d_model)
        x = x + attn.o_proj(y)

        h2 = blk.ln_mlp(x)
        x = x + blk.mlp(h2)
        del kk, vv, scores, y, q, k, v, qkv

    x = model.ln_final(x)
    if model.lm_head is not None:
        logits = model.lm_head(x)
    else:
        logits = matmul(x.reshape(B * T, cfg.d_model), model.tok_emb.weight.transpose())
        logits = logits.reshape(B, T, cfg.vocab_size)
    cache.length = pos0 + T
    return logits.data


def generate_with_cache(
    model: BoxzLM,
    prompt_ids: List[int],
    cfg: GenerationConfig,
    sampler,
    stop_ids: Optional[List[int]] = None,
    cache: Optional[KVCache] = None,
    on_token=None,
) -> "GenerationResultLite":
    """Sample ``cfg.max_new_tokens`` tokens using an incremental KV cache."""
    from .generator import GenerationResultLite

    t0 = __import__("time").time()
    ctx = model.cfg.context_length
    if cache is None:
        cache = KVCache(model.cfg.n_layer, model.cfg.n_head, model.cfg.head_dim, ctx)
    ids = [int(i) for i in prompt_ids]
    stops = set(int(s) for s in (stop_ids or []))
    rng = np.random.default_rng(cfg.seed)
    model.eval()
    produced = 0
    finish = "length"

    with no_grad():
        # 1. prime the cache with the prompt (all positions at once)
        window = ids[-ctx:]
        cache.reset()
        logits = _forward_one_step(model, np.asarray([window], dtype=np.int64), cache)
        for _ in range(int(cfg.max_new_tokens)):
            step_logits = logits[0, -1]
            recent = ids[-cfg.repetition_window:] if cfg.repetition_penalty != 1.0 else None
            nxt = sampler(step_logits, recent, rng)
            ids.append(nxt)
            produced += 1
            if on_token is not None:
                on_token(nxt)
            if nxt in stops:
                finish = "stop"
                break
            if cache.length >= ctx:
                # Context is full: drop the oldest half and re-prime the cache.
                keep = ids[-(ctx // 2):]
                cache.reset()
                logits = _forward_one_step(model, np.asarray([keep], dtype=np.int64), cache)
                # sliding context: subsequent decode continues from here
                ids_for_cache = keep
                continue
            logits = _forward_one_step(model, np.asarray([[nxt]], dtype=np.int64), cache)

    elapsed = __import__("time").time() - t0
    new_ids = ids[len(prompt_ids):]
    return GenerationResultLite(
        ids=new_ids, prompt_ids=list(prompt_ids), tokens_generated=produced,
        seconds=elapsed, finish_reason=finish,
        per_token_seconds=elapsed / max(1, produced),
    )

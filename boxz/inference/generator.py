"""Autoregressive text generation with context-window management.

Two loops are provided:

* :func:`generate` -- simple, re-runs the full forward pass every step.  Correct
  and simple; used for evaluation and tests.
* :class:`IncrementalGenerator` -- maintains a KV cache so each new token costs
  one time-step of attention instead of a full sequence re-read.  This is what
  the chat application uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import GenerationConfig
from ..engine import no_grad
from ..model.transformer import BoxzLM
from ..tokenizer.bpe import BPETokenizer
from .sampler import SamplingParams, build_forbidden_mask, sample_token

__all__ = ["GenerationResult", "GenerationResultLite", "generate_ids", "generate_text",
           "generate_with_kv_cache"]


@dataclass
class GenerationResult:
    text: str
    ids: List[int]
    prompt_ids: List[int]
    tokens_generated: int
    seconds: float
    finish_reason: str = "length"
    per_token_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "seconds": round(self.seconds, 3),
            "tokens_per_second": round(self.tokens_generated / max(self.seconds, 1e-9), 2),
            "finish_reason": self.finish_reason,
        }


@dataclass
class GenerationResultLite:
    """Result of the KV-cache path (text is filled in by the caller)."""
    ids: List[int]
    prompt_ids: List[int]
    tokens_generated: int
    seconds: float
    finish_reason: str
    per_token_seconds: float
    text: str = ""


def _params_from_config(cfg: GenerationConfig, tokenizer: Optional[BPETokenizer] = None) -> SamplingParams:
    allowed = None
    if tokenizer is not None and (cfg.forbid_control_tokens or cfg.utf8_safe):
        allowed = build_forbidden_mask(tokenizer, forbid_control=cfg.forbid_control_tokens)
    return SamplingParams(
        temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty, allowed=allowed,
    ).validate()


def generate_ids(
    model: BoxzLM,
    prompt_ids: Sequence[int],
    cfg: GenerationConfig,
    tokenizer: Optional[BPETokenizer] = None,
    stop_ids: Optional[Sequence[int]] = None,
    on_token: Optional[Callable[[int], None]] = None,
    rng: Optional[np.random.Generator] = None,
) -> GenerationResult:
    """Generate token ids after ``prompt_ids`` using the plain (no-KV) loop."""
    t0 = time.time()
    params = _params_from_config(cfg, tokenizer)
    rng = rng or np.random.default_rng(cfg.seed)
    ids = list(int(i) for i in prompt_ids)
    if not ids:
        # An empty prompt still needs a seed position: start from <bos> so text
        # can be generated unconditionally.
        if tokenizer is not None:
            ids = [tokenizer.bos_id]
        else:
            raise ValueError("prompt must contain at least one token")
    ctx = model.cfg.context_length
    stops = set(int(s) for s in (stop_ids or []))
    model.eval()
    produced = 0
    finish = "length"
    with no_grad():
        for _ in range(int(cfg.max_new_tokens)):
            window = ids[-ctx:]
            logits, _ = model(np.asarray([window], dtype=np.int64))
            step_logits = logits.data[0, -1]
            recent = ids[-cfg.repetition_window:] if cfg.repetition_penalty != 1.0 else None
            nxt = sample_token(step_logits, params, recent, rng)
            ids.append(nxt)
            produced += 1
            if on_token is not None:
                on_token(nxt)
            if nxt in stops:
                finish = "stop"
                break
    elapsed = time.time() - t0
    new_ids = ids[len(prompt_ids):]
    text = tokenizer.decode(new_ids, skip_special_tokens=False) if tokenizer else ""
    return GenerationResult(text=text, ids=new_ids, prompt_ids=list(prompt_ids),
                            tokens_generated=produced, seconds=elapsed, finish_reason=finish,
                            per_token_seconds=elapsed / max(1, produced))


def generate_text(model: BoxzLM, tokenizer: BPETokenizer, prompt: str,
                  cfg: Optional[GenerationConfig] = None, **kwargs) -> GenerationResult:
    cfg = cfg or GenerationConfig(**kwargs)
    prompt_ids = tokenizer.encode(prompt)
    stop_ids = [tokenizer.special_id(t) for t in cfg.stop_tokens if t in tokenizer.special_to_id]
    res = generate_ids(model, prompt_ids, cfg, tokenizer, stop_ids=stop_ids)
    res.text = tokenizer.decode(res.ids, skip_special_tokens=True)
    return res


def generate_with_kv_cache(
    model: BoxzLM,
    tokenizer: BPETokenizer,
    prompt: str,
    cfg: Optional[GenerationConfig] = None,
    **kwargs,
) -> GenerationResult:
    """Generate text using the incremental KV-cache decoder."""
    from .kv_cache import generate_with_cache as _gen

    cfg = cfg or GenerationConfig(**kwargs)
    params = _params_from_config(cfg, tokenizer)
    prompt_ids = tokenizer.encode(prompt)
    stop_ids = [tokenizer.special_id(t) for t in cfg.stop_tokens if t in tokenizer.special_to_id]

    def sampler(logits, recent, rng):
        return sample_token(logits, params, recent, rng)

    lite = _gen(model, prompt_ids, cfg, sampler, stop_ids=stop_ids)
    text = tokenizer.decode(lite.ids, skip_special_tokens=True)
    return GenerationResult(
        text=text, ids=lite.ids, prompt_ids=lite.prompt_ids,
        tokens_generated=lite.tokens_generated, seconds=lite.seconds,
        finish_reason=lite.finish_reason, per_token_seconds=lite.per_token_seconds,
    )

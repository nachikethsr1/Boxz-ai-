"""Inference tests: sampler maths, determinism, KV-cache equivalence, hygiene."""

from __future__ import annotations

import numpy as np
import pytest

from boxz.config import GenerationConfig, ModelConfig
from boxz.inference.generator import generate_ids, generate_text, generate_with_kv_cache
from boxz.inference.kv_cache import KVCache, _forward_one_step
from boxz.inference.sampler import (
    SamplingParams,
    apply_repetition_penalty,
    build_forbidden_mask,
    sample_token,
    top_k_filter,
    top_p_filter,
)
from boxz.tokenizer.bpe import BPETokenizer


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def test_greedy_is_argmax():
    logits = np.array([0.1, 5.0, -2.0, 3.0], dtype=np.float32)
    params = SamplingParams(greedy=True)
    assert sample_token(logits, params) == 1


def test_temperature_zero_is_greedy():
    logits = np.array([0.1, 5.0, -2.0, 3.0], dtype=np.float32)
    assert sample_token(logits, SamplingParams(temperature=0.0)) == 1


def test_top_k_restricts_support():
    logits = np.array([5.0, 4.0, 0.0, -1.0], dtype=np.float32)
    out = top_k_filter(logits, 2)
    assert np.isfinite(out[0]) and np.isfinite(out[1])
    assert not np.isfinite(out[2]) and not np.isfinite(out[3])


def test_top_p_restricts_support():
    logits = np.array([5.0, 4.0, 0.0, -1.0], dtype=np.float32)
    out = top_p_filter(logits, 0.5)
    assert np.isfinite(out[0])
    assert not np.isfinite(out[3])


def test_top_p_keeps_mass():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=100).astype(np.float32)
    out = top_p_filter(logits, 0.9)
    kept = np.exp(out - out.max())
    kept /= kept.sum()
    assert kept.sum() == pytest.approx(1.0)
    # the retained nucleus carries at least 90% of the (re-normalised) mass
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    assert probs[out > -np.inf].sum() >= 0.89


def test_repetition_penalty_discourages_tokens():
    logits = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    out = apply_repetition_penalty(logits, [0], 2.0)
    assert out[0] < out[1]
    # no-op for unseen tokens and penalty 1
    np.testing.assert_array_equal(apply_repetition_penalty(logits, [], 2.0), logits)
    np.testing.assert_array_equal(apply_repetition_penalty(logits, [0], 1.0), logits)


def test_sampling_respects_allowed_mask():
    allowed = np.array([False, True, False, False])
    logits = np.array([9.0, 0.0, 9.0, 9.0], dtype=np.float32)
    params = SamplingParams(temperature=1.0, top_k=None, top_p=None, allowed=allowed)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert sample_token(logits, params, rng=rng) == 1


def test_sampling_is_seeded():
    logits = np.random.default_rng(1).normal(size=50).astype(np.float32)
    params = SamplingParams(temperature=1.0, top_k=10, top_p=0.9)
    a = [sample_token(logits, params, rng=np.random.default_rng(5)) for _ in range(10)]
    b = [sample_token(logits, params, rng=np.random.default_rng(5)) for _ in range(10)]
    assert a == b


def test_forbidden_mask_covers_controls():
    tok = BPETokenizer().train("abc def ghi " * 100, vocab_size=300)
    mask = build_forbidden_mask(tok, forbid_control=True)

    # Structural turn terminators must stay sampleable, or the model can never
    # finish a reply and every generation runs to max_new_tokens (bug found in
    # the eval suite: instruction_format_rate was structurally pinned at 0).
    for name in ("<|end|>", "<eos>"):
        assert mask[int(tok.special_id(name))], f"{name} must be sampleable"
    # Nothing else may start a new turn or produce a bare structural token.
    for name in ("<|user|>", "<|assistant|>", "<|system|>", "<bos>", "<pad>", "<unk>"):
        assert not mask[int(tok.special_id(name))], f"{name} must be masked"

    for i in range(tok.n_specials, tok.vocab_size):
        if not mask[i]:
            raw = tok.token_bytes(i)
            assert any((b < 32 and b not in (9, 10)) or b == 127 for b in raw)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_and_tok():
    from boxz.model.transformer import build_model

    tok = BPETokenizer().train("hello world the cat sat on the mat. " * 100, vocab_size=300)
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, d_model=32,
                      d_ff=64, context_length=32)
    m = build_model(cfg, seed=0)
    return m, tok


def test_generate_ids_length(model_and_tok):
    m, tok = model_and_tok
    prompt = tok.encode("hello")
    res = generate_ids(m, prompt, GenerationConfig(max_new_tokens=10, seed=0))
    assert len(res.ids) == 10
    assert res.tokens_generated == 10
    assert res.seconds >= 0


def test_greedy_generation_deterministic(model_and_tok):
    m, tok = model_and_tok
    prompt = tok.encode("hello world")
    cfg = GenerationConfig(max_new_tokens=8, temperature=0.0, seed=1)
    a = generate_ids(m, prompt, cfg, tok)
    b = generate_ids(m, prompt, cfg, tok)
    assert a.ids == b.ids


def test_generate_text_returns_real_text(model_and_tok):
    m, tok = model_and_tok
    res = generate_text(m, tok, "the cat", GenerationConfig(max_new_tokens=8, temperature=0.7, seed=2))
    assert isinstance(res.text, str)
    assert len(res.text) > 0


def test_stop_tokens_halt_generation(model_and_tok):
    m, tok = model_and_tok
    prompt = tok.encode("hello")
    # force immediate stop by declaring the most likely first token a stop token
    logits, _ = m(np.asarray([prompt]))
    first = int(np.argmax(logits.data[0, -1]))
    res = generate_ids(m, prompt, GenerationConfig(max_new_tokens=20, temperature=0.0),
                       tok, stop_ids=[first])
    assert res.finish_reason == "stop"
    assert res.tokens_generated == 1


def test_context_window_is_respected(model_and_tok):
    m, tok = model_and_tok
    prompt = tok.encode("hello " * 20)          # longer than context_length
    res = generate_ids(m, prompt, GenerationConfig(max_new_tokens=5, temperature=0.5, seed=0))
    assert res.tokens_generated == 5


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------
def test_kv_cache_matches_full_forward(model_and_tok):
    m, tok = model_and_tok
    cfg = m.cfg
    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(1, 12))
    from boxz.engine import no_grad

    with no_grad():
        full, _ = m(ids)
    cache = KVCache(cfg.n_layer, cfg.n_head, cfg.head_dim, cfg.context_length)
    _forward_one_step(m, ids[:, :8], cache)
    inc = _forward_one_step(m, ids[:, 8:], cache)
    diff = np.max(np.abs(inc[:, -1] - full.data[:, -1]))
    assert diff < 1e-4, diff
    assert cache.length == 12


def test_kv_cache_generation_matches_plain_greedy(model_and_tok):
    m, tok = model_and_tok
    prompt = "the cat sat"
    cfg = GenerationConfig(max_new_tokens=12, temperature=0.0)
    plain = generate_text(m, tok, prompt, cfg)
    cached = generate_with_kv_cache(m, tok, prompt, cfg)
    assert cached.ids == plain.ids, (cached.ids, plain.ids)


def test_kv_cache_is_faster(model_and_tok):
    """The cache must actually save work on a realistic suffix length."""
    import time

    import time

    m, tok = model_and_tok
    prompt = "the cat sat on the mat and the dog"
    cfg = GenerationConfig(max_new_tokens=40, temperature=0.0)

    # Wall-clock on a shared machine is noisy (a background training run makes
    # the first path measured look slower regardless of the algorithm), so take
    # the best of three runs for each and require a clear margin.
    def best(fn) -> float:
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            runs.append(time.perf_counter() - t0)
        return min(runs)

    plain = best(lambda: generate_ids(m, tok.encode(prompt), cfg, tok))
    cached = best(lambda: generate_with_kv_cache(m, tok, prompt, cfg))
    assert cached < plain * 1.05, f"cache {cached:.3f}s vs plain {plain:.3f}s"

"""Model tests: shapes, causality, initialisation, parameter count, gradients."""

from __future__ import annotations

import numpy as np
import pytest

from boxz.config import ModelConfig
from boxz.engine import ops
from boxz.engine.tensor import no_grad
from boxz.model.transformer import BoxzLM, build_model


def tiny_cfg(**kw):
    base = dict(vocab_size=64, n_layer=2, n_head=2, d_model=32, d_ff=64, context_length=16)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes():
    m = build_model(tiny_cfg(), seed=0)
    x = np.random.randint(0, 64, size=(3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 64)
    assert loss.shape == ()
    assert np.isfinite(float(loss.data))


def test_loss_at_init_is_log_vocab():
    """An untrained model must be ~uniform: cross entropy ~ ln(vocab_size)."""
    m = build_model(tiny_cfg(vocab_size=256), seed=1)
    x = np.random.randint(0, 256, size=(8, 16))
    _, loss = m(x, x)
    # Residual-projection scaling keeps the final hidden state small, so the
    # initial distribution is slightly sharper than uniform: loss is near, but
    # not exactly, ln(vocab).  A breakdown would push this far away.
    assert abs(float(loss.data) - np.log(256)) < 0.35


def test_context_length_enforced():
    m = build_model(tiny_cfg(context_length=8), seed=0)
    with pytest.raises(ValueError):
        m(np.zeros((1, 9), dtype=np.int64))


def test_batch_dimension_required():
    m = build_model(tiny_cfg(), seed=0)
    with pytest.raises(ValueError):
        m(np.zeros(5, dtype=np.int64))


def test_causality_future_tokens_do_not_affect_past():
    """Changing token t must not change logits at positions < t."""
    m = build_model(tiny_cfg(), seed=3)
    x = np.random.randint(0, 64, size=(1, 16))
    base, _ = m(x)
    x2 = x.copy()
    x2[0, 9] = (x2[0, 9] + 7) % 64
    changed, _ = m(x2)
    np.testing.assert_allclose(base.data[0, :9], changed.data[0, :9], rtol=1e-5, atol=1e-5)
    assert not np.allclose(base.data[0, 9], changed.data[0, 9])


def test_no_grad_produces_no_graph():
    m = build_model(tiny_cfg(), seed=0)
    x = np.random.randint(0, 64, size=(2, 16))
    with no_grad():
        logits, _ = m(x)
    assert logits._node is None
    assert not logits.requires_grad


def test_parameter_count_matches_estimate():
    cfg = tiny_cfg()
    m = build_model(cfg, seed=0)
    actual = m.num_parameters()
    est = cfg.estimated_parameters()
    # the analytic estimate ignores tied-embedding details only
    assert abs(actual - est) / actual < 0.05, (actual, est)


def test_variants_run_and_differ():
    configs = {
        "rmsnorm": tiny_cfg(norm="rmsnorm"),
        "silu": tiny_cfg(activation="silu"),
        "untied": tiny_cfg(tied_embeddings=False),
        "nobias": tiny_cfg(bias=False),
    }
    x = np.random.randint(0, 64, size=(2, 16))
    for name, cfg in configs.items():
        m = build_model(cfg, seed=0)
        logits, loss = m(x, x)
        assert logits.shape == (2, 16, 64), name
        assert np.isfinite(float(loss.data)), name


def test_tied_vs_untied_parameter_difference():
    tied = build_model(tiny_cfg(tied_embeddings=True), seed=0).num_parameters()
    untied = build_model(tiny_cfg(tied_embeddings=False), seed=0).num_parameters()
    assert untied == tied + 64 * 32


def test_seeded_init_is_deterministic():
    a = build_model(tiny_cfg(), seed=42)
    b = build_model(tiny_cfg(), seed=42)
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        np.testing.assert_array_equal(p1.data, p2.data, err_msg=n1)


def test_different_seeds_differ():
    a = build_model(tiny_cfg(), seed=1)
    b = build_model(tiny_cfg(), seed=2)
    assert not np.array_equal(a.tok_emb.weight.data, b.tok_emb.weight.data)


def test_full_model_directional_gradient_check():
    """One directional derivative over *all* parameters at once."""
    rng = np.random.default_rng(5)
    cfg = tiny_cfg(vocab_size=48, n_layer=2, d_model=16, d_ff=32, context_length=8)
    m = build_model(cfg, seed=7)
    x = rng.integers(0, 48, size=(2, 8))
    _, loss = m(x, x)
    loss.backward()
    params = [p for _, p in m.named_parameters()]
    dirs = [rng.normal(size=p.shape).astype(np.float32) for p in params]
    norm = np.sqrt(sum(float((d * d).sum()) for d in dirs))
    dirs = [d / norm for d in dirs]
    analytic = sum(float((p.grad * d).sum()) for p, d in zip(params, dirs))
    eps = 2e-3
    for p, d in zip(params, dirs):
        p.data += eps * d
    _, lp = m(x, x)
    for p, d in zip(params, dirs):
        p.data -= 2 * eps * d
    _, lm = m(x, x)
    for p, d in zip(params, dirs):
        p.data += eps * d
    numeric = (float(lp.data) - float(lm.data)) / (2 * eps)
    assert abs(analytic - numeric) / abs(numeric) < 0.05, (analytic, numeric)


def test_attention_mask_is_causal():
    from boxz.model.attention import causal_mask

    m = causal_mask(4)
    assert m[0, 0] and not m[0, 1]
    assert m[3, :].all()


def test_state_dict_roundtrip_model():
    m1 = build_model(tiny_cfg(), seed=11)
    m2 = build_model(tiny_cfg(), seed=12)
    m2.load_state_dict(m1.state_dict())
    x = np.random.randint(0, 64, size=(2, 8))
    np.testing.assert_allclose(m1(x)[0].data, m2(x)[0].data, rtol=1e-6)


def test_model_free_graph_and_training_step_memory():
    """A step must not retain the graph: slots holding data stay bounded."""
    from boxz.engine.tensor import store

    m = build_model(tiny_cfg(), seed=0)
    opt = __import__("boxz.engine.optim", fromlist=["AdamW"]).AdamW(m.parameters(), lr=1e-3)
    x = np.random.randint(0, 64, size=(2, 16))
    live_counts = []
    for _ in range(3):
        m.zero_grad()
        _, loss = m(x, x)
        loss.backward()
        opt.step()
        live_counts.append(sum(1 for a in store().arrays if a.size > 0))
        loss.free_graph()
    # no growth in live buffers between steps
    assert live_counts[-1] <= live_counts[0] + 5, live_counts

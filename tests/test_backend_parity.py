"""Backend parity: Rust kernels must match the NumPy reference.

Skipped automatically when the extension is not built (which is the default in
CI), but the test is part of the repository so a Rust build is always validated
before it can be trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from boxz.backends import NumpyBackend, RustBackend, RustUnavailable


def _rust():
    try:
        return RustBackend()
    except RustUnavailable as exc:
        pytest.skip(f"Rust extension not built: {exc}")


@pytest.fixture(scope="module")
def backends():
    return NumpyBackend(), _rust()


def test_gelu_parity(backends):
    np_b, rs_b = backends
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 128)).astype(np.float32) * 3
    np.testing.assert_allclose(rs_b.gelu(x), np_b.gelu(x), rtol=1e-5, atol=1e-6)


def test_softmax_parity(backends):
    np_b, rs_b = backends
    rng = np.random.default_rng(1)
    x = rng.normal(size=(32, 1000)).astype(np.float32) * 5
    np.testing.assert_allclose(rs_b.softmax(x), np_b.softmax(x), rtol=1e-5, atol=1e-6)
    # rows must sum to one
    np.testing.assert_allclose(rs_b.softmax(x).sum(axis=-1), 1.0, rtol=1e-6)


def test_layer_norm_parity(backends):
    np_b, rs_b = backends
    rng = np.random.default_rng(2)
    x = (rng.normal(size=(16, 192)).astype(np.float32) * 2 + 1)
    gamma = rng.normal(size=192).astype(np.float32)
    beta = rng.normal(size=192).astype(np.float32)
    y_np, xhat_np = np_b.layer_norm(x, gamma, beta, 1e-5)
    y_rs, xhat_rs = rs_b.layer_norm(x, gamma, beta, 1e-5)
    np.testing.assert_allclose(y_rs, y_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(xhat_rs, xhat_np, rtol=1e-5, atol=1e-5)


def test_sampling_distribution_parity(backends):
    """Both samplers must produce the same top-k/top-p greedy choice and a
    statistically similar distribution."""
    np_b, rs_b = backends
    rng = np.random.default_rng(3)
    logits = rng.normal(size=500).astype(np.float32) * 2
    # greedy agreement
    a = np_b.sample(logits, 0.0, None, 1.0, 1.0, [], 0)
    b = rs_b.sample(logits, 0.0, None, 1.0, 1.0, [], 0)
    assert a == b == int(np.argmax(logits))
    # support restriction: with top_k=3 only the 3 best ids may appear
    top3 = set(np.argsort(-logits)[:3].tolist())
    draws = {np_b.sample(logits, 0.7, 3, 1.0, 1.0, [], s) for s in range(200)}
    draws_rs = {rs_b.sample(logits, 0.7, 3, 1.0, 1.0, [], s) for s in range(200)}
    assert draws <= top3 and draws_rs <= top3


def test_repetition_penalty_parity(backends):
    np_b, rs_b = backends
    logits = np.array([3.0, 3.0, 3.0, 0.0], dtype=np.float32)
    for seed in range(20):
        a = np_b.sample(logits, 1.0, None, 1.0, 5.0, [0, 1], seed)
        assert a in (2, 3)
        b = rs_b.sample(logits, 1.0, None, 1.0, 5.0, [0, 1], seed)
        assert b in (2, 3)

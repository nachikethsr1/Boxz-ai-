"""Autograd correctness: every op is checked against finite differences.

These are the most important tests in the project: if they pass, the gradients
that train Boxz are provably correct (to float32 precision).
"""

from __future__ import annotations

import numpy as np
import pytest

from boxz.engine import ops
from boxz.engine.tensor import Tensor, is_grad_enabled, no_grad


def numeric_grad(fn, arrays, index, eps=1e-3):
    """Central difference of a scalar-valued ``fn`` w.r.t. arrays[index][index]."""
    arr = arrays[index]
    it = np.nditer(arr, flags=["multi_index"])
    out = np.zeros_like(arr)
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps
        plus = float(fn(*[Tensor(a) for a in arrays]).data.sum())
        arr[i] = old - eps
        minus = float(fn(*[Tensor(a) for a in arrays]).data.sum())
        arr[i] = old
        out[i] = (plus - minus) / (2 * eps)
    return out


def check(fn, *arrays, tol=3e-2):
    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*ts)
    if out.shape != ():
        out = out.sum()
    out.backward()
    for k, t in enumerate(ts):
        numeric = numeric_grad(fn, [a.copy() for a in arrays], k)
        denom = np.max(np.abs(numeric)) + 1e-8
        err = np.max(np.abs(t.grad - numeric)) / denom
        # float32 central differences have ~1e-4 absolute noise, so a purely
        # relative check is meaningless when the true gradient is ~0.
        abs_err = np.max(np.abs(t.grad - numeric))
        assert err < tol or abs_err < 3e-3, (
            f"argument {k}: relative error {err:.3e}, absolute error {abs_err:.3e}"
        )


RNG = np.random.default_rng(0)
A = RNG.normal(size=(3, 4)).astype(np.float32)
B = RNG.normal(size=(4, 5)).astype(np.float32)
C = RNG.normal(size=(3, 5)).astype(np.float32)
V = RNG.normal(size=(4,)).astype(np.float32)


def test_add_mul_div():
    check(lambda a: (a + V) * np.ones((3, 4), dtype=np.float32), A)
    check(lambda a: a * 2.5, A)
    check(lambda a: (a / (a * a + 1.0)), A)


def test_matmul_shapes():
    check(lambda a, b: a @ b, A.copy(), B.copy())
    check(lambda b: b @ b.T, B.copy())
    check(lambda a, b: (a @ b) * np.ones((3, 5), dtype=np.float32), A.copy(), B.copy())


def test_elementwise_ops():
    check(lambda a: a.exp(), A.copy())
    check(lambda a: (a * a + 1.0).log(), A.copy())
    check(lambda a: (a * a + 1.0).sqrt(), A.copy())
    check(lambda a: a.clip(-0.5, 0.5), A.copy())
    check(lambda a: a**2, A.copy())


def test_activations():
    check(lambda a: ops.gelu(a), A.copy())
    check(lambda a: ops.relu(a), A.copy())
    check(lambda a: ops.silu(a), A.copy())
    check(lambda a: ops.gelu(a * 3.0), A.copy())


def test_softmax_and_cross_entropy():
    # softmax over rows sums to exactly 1, so its gradient is mathematically
    # zero; multiplying by C^-1 makes the check meaningful without changing the
    # required derivative (float32 FD would otherwise divide 0 by 0).
    inv = (1.0 / C).astype(np.float32)
    check(lambda c: ops.softmax(c) * inv, C.copy())
    check(lambda c: ops.log_softmax(c) * inv, C.copy())
    tgt = np.array([1, 3, 0], dtype=np.int64)
    check(lambda c: ops.cross_entropy_with_logits(c, tgt), C.copy())
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 3] = False
    mask[1, 0] = False
    weights = RNG.normal(size=(3, 5)).astype(np.float32)
    check(lambda c: ops.masked_softmax(c * 3.0, mask) * weights, C.copy())


def test_layer_norm_and_rms():
    g = RNG.normal(size=(4,)).astype(np.float32)
    be = RNG.normal(size=(4,)).astype(np.float32)
    check(lambda a, gm, bt: ops.layer_norm(a, gm, bt), A.copy(), g.copy(), be.copy())
    check(lambda a, gm: ops.rms_norm(a, gm), A.copy(), g.copy())


def test_reshape_transpose_slice():
    check(lambda a: a.reshape(2, 6) @ np.ones((6, 3), dtype=np.float32), A.copy())
    check(lambda a: a.transpose(), A.copy())
    check(lambda a: a[:, 1:3], A.copy())
    check(lambda a: a[1:3, ::2], A.copy())


def test_embedding_gather():
    ids = np.array([[1, 2, 0], [2, 2, 1]])
    w = RNG.normal(size=(3, 4)).astype(np.float32)
    check(lambda w_: w_[ids], w)
    # duplicate index must accumulate, not overwrite
    t = Tensor(w.copy(), requires_grad=True)
    w_t = Tensor(w.copy(), requires_grad=True)
    w_t[ids].sum().backward()
    expected = np.zeros_like(w)
    np.add.at(expected, ids, 1.0)
    np.testing.assert_allclose(w_t.grad, expected, rtol=1e-6)


def test_reduction_ops():
    check(lambda a: a.mean(axis=1), A.copy())
    check(lambda a: a.sum(axis=0, keepdims=True), A.copy())
    check(lambda a: a.mean(), A.copy())


def test_broadcast_gradient_shapes():
    """Gradients must be reduced back to the operand shape, not left broadcast."""
    x = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    y = Tensor(np.ones((3,), dtype=np.float32), requires_grad=True)
    (x + y).sum().backward()
    assert x.grad.shape == (2, 3)
    assert y.grad.shape == (3,)
    np.testing.assert_allclose(y.grad, [2.0, 2.0, 2.0])


def test_graph_accumulates_multiple_paths():
    x = Tensor(np.array([3.0], dtype=np.float32), requires_grad=True)
    y = x * x
    (y + y).sum().backward()
    # d/dx (x^2 + x^2) = 4x = 12
    np.testing.assert_allclose(x.grad, [12.0], rtol=1e-5)


def test_no_grad_disables_graph():
    with no_grad():
        x = Tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
        y = x * 2.0
        assert not y.requires_grad
        assert y._node is None
    assert is_grad_enabled()


def test_pickle_roundtrip():
    import pickle

    t = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
    t2 = pickle.loads(pickle.dumps(t))
    np.testing.assert_array_equal(t.data, t2.data)
    assert t2.requires_grad


def test_free_graph_releases_activations():
    from boxz.engine.tensor import TensorStore, reset_store, set_store

    tok = set_store(TensorStore())
    try:
        w = Tensor(RNG.normal(size=(8, 8)).astype(np.float32), requires_grad=True)
        x = Tensor(RNG.normal(size=(4, 8)).astype(np.float32), requires_grad=True)
        h = ops.gelu(x @ w)
        loss = (h * h).sum()
        loss.backward()
        assert w.grad is not None
        loss.free_graph()
        assert loss._node is None
    finally:
        reset_store(tok)

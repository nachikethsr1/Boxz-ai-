"""Layer-level tests: shapes, initialisation, state dicts, optimiser maths."""

from __future__ import annotations

import numpy as np
import pytest

from boxz.engine.layers import Dropout, Embedding, LayerNorm, Linear, Module, RMSNorm
from boxz.engine.optim import AdamW, LRSchedule, SGD, clip_grad_norm_
from boxz.engine.tensor import Tensor


def test_linear_shapes_and_bias_free():
    lin = Linear(8, 4, bias=True)
    out = lin(Tensor(np.random.randn(2, 3, 8).astype(np.float32)))
    assert out.shape == (2, 3, 4)
    nobias = Linear(8, 4, bias=False)
    assert nobias._parameters.get("bias") is None
    assert nobias(Tensor(np.random.randn(2, 8).astype(np.float32))).shape == (2, 4)


def test_linear_weight_init_scale():
    lin = Linear(256, 64, std=0.05)
    assert abs(float(lin.weight.data.std()) - 0.05) < 0.01


def test_layernorm_properties():
    ln = LayerNorm(16)
    x = np.random.randn(3, 5, 16).astype(np.float32) * 4 + 2
    y = ln(Tensor(x)).data
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(y.std(axis=-1), 1.0, atol=1e-3)
    # with learned gamma/beta the output follows them
    ln.gamma.data = np.full(16, 2.0, dtype=np.float32)
    ln.beta.data = np.full(16, 3.0, dtype=np.float32)
    y2 = ln(Tensor(x)).data
    np.testing.assert_allclose(y2.mean(axis=-1), 3.0, atol=1e-4)


def test_rmsnorm_scale():
    rn = RMSNorm(16)
    x = np.random.randn(4, 16).astype(np.float32)
    y = rn(Tensor(x)).data
    rms = np.sqrt((y * y).mean(axis=-1))
    np.testing.assert_allclose(rms, 1.0, atol=1e-4)


def test_embedding_lookup():
    emb = Embedding(10, 4)
    out = emb(np.array([[1, 2], [3, 3]]))
    assert out.shape == (2, 2, 4)
    np.testing.assert_array_equal(out.data[0, 0], emb.weight.data[1])


def test_dropout_train_eval_difference():
    d = Dropout(0.5)
    d.seed(0)
    x = Tensor(np.ones((1000,), dtype=np.float32))
    train_out = d(x).data
    assert train_out.mean() == pytest.approx(1.0, abs=0.1)   # inverted dropout is unbiased
    assert (train_out == 0).mean() > 0.4
    d.eval()
    np.testing.assert_array_equal(d(x).data, np.ones((1000,)))


def test_state_dict_roundtrip():
    m = Linear(4, 3)
    import copy

    state = m.state_dict()
    original = copy.deepcopy(state)
    for k in state:
        state[k][...] = 0
    m.load_state_dict(original)
    for k, v in m.state_dict().items():
        np.testing.assert_array_equal(v, original[k])


def test_state_dict_strict_errors():
    m = Linear(4, 3)
    with pytest.raises(KeyError):
        m.load_state_dict({"weight": np.zeros((3, 4), dtype=np.float32)})


def test_optimizer_reduces_quadratic():
    target = np.array([2.0, -3.0], dtype=np.float32)
    p = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)

    class P:  # minimal Parameter-compatible shim
        pass

    for cls in (SGD, AdamW):
        p = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
        opt = cls([p], lr=0.1)  # type: ignore[arg-type]
        for _ in range(300):
            p.grad = 2 * (p.data - target)
            opt.step()
        assert np.allclose(p.data, target, atol=1e-2)


def test_grad_clipping():
    p = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
    p.grad = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    norm = clip_grad_norm_([p], max_norm=1.0)  # type: ignore[arg-type]
    assert norm == pytest.approx(5.0)
    assert np.linalg.norm(p.grad) == pytest.approx(1.0, rel=1e-5)


def test_lr_schedule_shape():
    s = LRSchedule(mode="cosine", lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert s.get(0) < s.get(5) < s.get(10)
    assert s.get(10) == pytest.approx(1.0, rel=1e-3)
    assert s.get(110) == pytest.approx(0.1, rel=1e-6)
    values = [s.get(i) for i in range(10, 111, 10)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))  # monotone decay


def test_adamw_weight_decay_applied():
    p = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.5, grad_clip=None)  # type: ignore[arg-type]
    p.grad = np.zeros(1, dtype=np.float32)
    before = float(p.data[0])
    opt.step()
    assert float(p.data[0]) < before


def test_module_parameter_enumeration():
    class Net(Module):
        def __init__(self):
            super().__init__()
            self.register_module("a", Linear(2, 3))
            self.register_module("b", Linear(3, 1))

    n = Net()
    names = [x for x, _ in n.named_parameters()]
    assert names == ["a.weight", "a.bias", "b.weight", "b.bias"]
    assert n.num_parameters() == 3 * 2 + 3 + 1 * 3 + 1

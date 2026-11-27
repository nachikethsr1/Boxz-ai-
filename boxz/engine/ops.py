"""Differentiable primitives used by the Boxz model (softmax, GELU, cross entropy...)."""

from __future__ import annotations

import numpy as np

from .tensor import Context, Function, Tensor, apply

__all__ = [
    "softmax",
    "log_softmax",
    "masked_softmax",
    "gelu",
    "relu",
    "silu",
    "cross_entropy_with_logits",
    "layer_norm",
    "rms_norm",
    "dropout_mask",
]

_EPS = 1e-5


class Softmax(Function):
    name = "softmax"

    def forward(self, ctx: Context, x: np.ndarray, axis: int) -> np.ndarray:  # type: ignore[override]
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        out = e / np.sum(e, axis=axis, keepdims=True)
        ctx.save(out=out, axis=axis)
        return out

    def backward(self, ctx: Context, g: np.ndarray, x: np.ndarray, axis: int):  # type: ignore[override]
        out = ctx.data["out"]
        ax = ctx.data["axis"]
        return (out * (g - np.sum(g * out, axis=ax, keepdims=True)), None)


class LogSoftmax(Function):
    name = "log_softmax"

    def forward(self, ctx: Context, x: np.ndarray, axis: int) -> np.ndarray:  # type: ignore[override]
        m = np.max(x, axis=axis, keepdims=True)
        z = x - m
        lse = np.log(np.sum(np.exp(z), axis=axis, keepdims=True))
        out = z - lse
        ctx.save(out=out, axis=axis)
        return out

    def backward(self, ctx: Context, g: np.ndarray, x: np.ndarray, axis: int):  # type: ignore[override]
        out = ctx.data["out"]
        ax = ctx.data["axis"]
        return (g - np.exp(out) * np.sum(g, axis=ax, keepdims=True), None)


class MaskedSoftmax(Function):
    """Softmax over ``axis`` with a boolean mask (masked positions get weight 0).

    ``mask`` must broadcast to the input shape.  Used for causal attention,
    where the mask is a constant ``j <= i`` lower-triangular matrix.

    Implementation notes (these matter for CPU performance):
    * masked positions receive ``-30`` *after* subtracting the row max instead of
      ``-inf`` before it.  ``-sqrt(2**127)`` is the smallest value for which
      ``exp`` still returns a normal float, so no subnormals are produced --
      subnormal-heavy ``exp`` is ~10x slower on x86.
    * softmax is invariant to adding a constant per row, but *not* to adding a
      different constant per element across the batch.  A per-element bias is
      fine for the backward pass (it only re-weights ``g``), which lets the
      forward path avoid materialising a second masked copy.
    """

    name = "masked_softmax"

    def forward(self, ctx: Context, x: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:  # type: ignore[override]
        ctx.args = (x, mask, axis)
        add_mask = np.where(mask, np.float32(0.0), np.float32(-30.0)).astype(np.float32, copy=False)
        xm = np.add(x, add_mask)
        mx = np.max(xm, axis=axis, keepdims=True)
        np.subtract(xm, mx, out=xm)
        np.exp(xm, out=xm)
        denom = np.sum(xm, axis=axis, keepdims=True)
        np.divide(xm, denom, out=xm)
        lse = np.log(denom) - mx  # recovered log-sum-exp (kept for the backward pass)
        ctx.save(out=xm, lse=lse, axis=axis)
        return xm

    def backward(self, ctx: Context, g: np.ndarray, x: np.ndarray, mask: np.ndarray, axis: int):  # type: ignore[override]
        out = ctx.data["out"]
        ax = ctx.data["axis"]
        if mask is None or np.size(mask) == 0:
            mask = ctx.args[1]
        if np.ndim(g) == 0:
            g = np.full_like(out, g)
        gx = g * out
        s = np.sum(gx, axis=ax, keepdims=True)
        gx = out * (g - s)
        gx = np.where(mask, gx, np.float32(0.0))
        return gx, None, None


class Gelu(Function):
    """Gaussian Error Linear Unit, tanh approximation (as used by GPT-2).

    y = 0.5 x (1 + tanh(c (x + 0.044715 x^3))),  c = sqrt(2/pi)

    ``x*x*x`` is ~1.5x faster than ``np.power(x, 3)`` and bit-identical in fp32.
    The forward keeps ``tanh(z)`` for the backward pass, which is where most of
    the arithmetic lives.
    """

    name = "gelu"

    def forward(self, ctx: Context, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        x3 = x * x * x
        z = np.float32(0.7978845608) * (x + np.float32(0.044715) * x3)
        t = np.tanh(z)
        ctx.save(x=x, t=t)
        return np.float32(0.5) * x * (np.float32(1.0) + t)

    def backward(self, ctx: Context, g: np.ndarray, x: np.ndarray):  # type: ignore[override]
        # Buffers saved in forward alias live tensors: never mutate them.
        #   dy/dx = 0.5 (1 + tanh z) + 0.5 x (1 - tanh^2 z) * dz/dx
        #   dz/dx = c (1 + 3 * 0.044715 x^2)
        x_ = ctx.data["x"]
        t_ = ctx.data["t"]
        dz = np.float32(0.7978845608) * (np.float32(1.0) + np.float32(0.134145) * x_ * x_)
        cdf = np.float32(0.5) * (np.float32(1.0) + t_)
        pdf = cdf * (np.float32(1.0) - t_) * dz
        return (g * (cdf + x_ * pdf),)


class Relu(Function):
    name = "relu"

    def forward(self, ctx: Context, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        ctx.save(mask=(x > 0).astype(np.float32))
        return np.maximum(x, 0.0)

    def backward(self, ctx: Context, g: np.ndarray, x: np.ndarray):  # type: ignore[override]
        return (g * ctx.data["mask"],)


class Silu(Function):
    name = "silu"

    def forward(self, ctx: Context, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        s = 1.0 / (1.0 + np.exp(-x))
        ctx.save(s=s, x=x)
        return x * s

    def backward(self, ctx: Context, g: np.ndarray, x: np.ndarray):  # type: ignore[override]
        # saved buffers are read-only; x comes from the caller
        s, x_ = ctx.data["s"], ctx.data["x"]
        return (g * (s + x_ * s * (1.0 - s)),)


class CrossEntropyWithLogits(Function):
    """Mean cross entropy of logits against integer token ids.

    Forward:  loss = mean_{i not ignored} -log softmax(logits_i)[target_i]
    Backward: d loss / d logits = (softmax(logits) - onehot(target)) / N

    The expensive part is the ``(batch*time, vocab)`` softmax, so it is done in
    place on a single temporary array (see MaskedSoftmax for the exp/subnormal
    reasoning).  Gradients only exist for the logits argument, and they are
    materialised lazily on the backward pass to avoid an extra 3MB allocation
    per step at vocab 4096.
    """

    name = "cross_entropy"

    def forward(self, ctx: Context, logits: np.ndarray, targets: np.ndarray, ignore_index: int):  # type: ignore[override]
        v = logits.shape[-1]
        tgt = targets.reshape(-1).astype(np.int64)
        valid = tgt != ignore_index
        n = int(valid.sum())
        if n == 0:
            ctx.save(empty=True)
            return np.zeros((), dtype=np.float32)
        rows = np.nonzero(valid)[0]
        sel = np.take(logits.reshape(-1, v), rows, axis=0)   # copy: (n, vocab)
        m = np.max(sel, axis=1, keepdims=True)
        sel -= m
        np.exp(sel, out=sel)
        denom = np.sum(sel, axis=1, keepdims=True)
        tgt_valid = tgt[rows]
        ar = np.arange(n)
        p_target = sel[ar, tgt_valid] / denom[:, 0]
        loss = -np.log(np.maximum(p_target, np.float32(1e-30)))
        ctx.save(shape=logits.shape, rows=rows, targets=tgt_valid, denom=denom, n=n, mean=loss.mean())
        return np.float32(loss.mean())

    def backward(self, ctx: Context, g: np.ndarray, logits: np.ndarray, targets: np.ndarray, ignore_index: int):  # type: ignore[override]
        if ctx.data.get("empty"):
            return np.zeros_like(logits), None, None
        shape = ctx.data["shape"]
        rows = ctx.data["rows"]
        tgt = ctx.data["targets"]
        denom = ctx.data["denom"]
        n = ctx.data["n"]
        v = shape[-1]
        grad = np.zeros(shape, dtype=np.float32)
        flat = grad.reshape(-1, v)
        sel = np.take(logits.reshape(-1, v), rows, axis=0)
        sel -= np.max(sel, axis=1, keepdims=True)
        np.exp(sel, out=sel)
        sel /= denom
        sel[np.arange(n), tgt] -= np.float32(1.0)
        flat[rows] = sel
        coef = np.float32(float(np.asarray(g, dtype=np.float64).reshape(())) / max(n, 1))
        np.multiply(grad, coef, out=grad)
        return grad, None, None


class LayerNormFn(Function):
    """Classic LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta."""

    name = "layer_norm"

    def forward(self, ctx: Context, x, gamma, beta, eps):  # type: ignore[override]
        mu = np.mean(x, axis=-1, keepdims=True)
        xc = x - mu
        var = np.mean(xc * xc, axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + np.float32(eps))
        xhat = xc * rstd
        ctx.save(xhat=xhat, rstd=rstd, gamma=gamma)
        return xhat * gamma + beta

    def backward(self, ctx: Context, g, x, gamma, beta, eps):  # type: ignore[override]
        xhat = ctx.data["xhat"]
        rstd = ctx.data["rstd"]
        gamma = ctx.data["gamma"]
        d = x.shape[-1]
        ggamma = np.sum(g * xhat, axis=tuple(range(g.ndim - 1)))
        gbeta = np.sum(g, axis=tuple(range(g.ndim - 1)))
        gxhat = g * gamma
        gx = (gxhat - np.mean(gxhat, axis=-1, keepdims=True)
              - xhat * np.mean(gxhat * xhat, axis=-1, keepdims=True)) * rstd
        return gx, ggamma, gbeta, None


class RmsNormFn(Function):
    """RMSNorm: y = x / rms(x) * gamma."""

    name = "rms_norm"

    def forward(self, ctx: Context, x, gamma, eps):  # type: ignore[override]
        ms = np.mean(x * x, axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(ms + np.float32(eps))
        xhat = x * rstd
        ctx.save(xhat=xhat, rstd=rstd, gamma=gamma)
        return xhat * gamma

    def backward(self, ctx: Context, g, x, gamma, eps):  # type: ignore[override]
        xhat, rstd, gamma = ctx.data["xhat"], ctx.data["rstd"], ctx.data["gamma"]
        ggamma = np.sum(g * xhat, axis=tuple(range(g.ndim - 1)))
        gxhat = g * gamma
        gx = rstd * (gxhat - xhat * np.mean(gxhat * xhat, axis=-1, keepdims=True))
        return gx, ggamma, None


softmax_fn = Softmax()
log_softmax_fn = LogSoftmax()
masked_softmax_fn = MaskedSoftmax()
gelu_fn = Gelu()
relu_fn = Relu()
silu_fn = Silu()
ce_fn = CrossEntropyWithLogits()
ln_fn = LayerNormFn()
rms_fn = RmsNormFn()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return softmax_fn(x, axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return log_softmax_fn(x, axis)


def masked_softmax(x: Tensor, mask, axis: int = -1) -> Tensor:
    return masked_softmax_fn(x, np.asarray(mask, dtype=bool), axis)


def gelu(x: Tensor) -> Tensor:
    return gelu_fn(x)


def relu(x: Tensor) -> Tensor:
    return relu_fn(x)


def silu(x: Tensor) -> Tensor:
    return silu_fn(x)


def cross_entropy_with_logits(logits: Tensor, targets, ignore_index: int = -1) -> Tensor:
    tgt = targets.data if isinstance(targets, Tensor) else np.asarray(targets, dtype=np.int64)
    return ce_fn(logits, tgt, ignore_index)


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = _EPS) -> Tensor:
    return ln_fn(x, gamma, beta, float(eps))


def rms_norm(x: Tensor, gamma: Tensor, eps: float = _EPS) -> Tensor:
    return rms_fn(x, gamma, float(eps))


def dropout_mask(shape, p: float, rng: np.random.Generator) -> np.ndarray:
    """Inverted-dropout multiplier."""
    if p <= 0.0:
        return np.ones(shape, dtype=np.float32)
    keep = rng.random(shape) >= p
    return (keep.astype(np.float32) / np.float32(1.0 - p))

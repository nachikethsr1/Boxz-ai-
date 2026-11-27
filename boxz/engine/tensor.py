"""Reverse-mode automatic differentiation on NumPy arrays.

This module is the numerical heart of Boxz.  Nothing in the project may depend
on PyTorch, TensorFlow, JAX or any other deep-learning framework: every gradient
that trains the language model is produced here and then verified against
finite differences in ``tests/test_grad.py``.

Design
------
* A :class:`Tensor` wraps a ``float32`` NumPy array.  Array-valued attributes
  (``data``, ``grad``) are int32 viewable handles into a :class:`TensorStore`.
  This keeps every Tensor cheap to copy, pickle and send across processes;
  the store is process-local and must be cloned when crossing process
  boundaries (see :meth:`TensorStore.clone_dense`).
* Only tensors with ``requires_grad=True`` that are involved in an operation
  keep a slot in the backward graph.  No-op wrappers (``requires_grad``
  bookkeeping, ``no_grad`` contexts) never create graph nodes.
* Broadcasting follows NumPy semantics; gradients are summed back to the
  original operand shape by :meth:`Tensor.backward`.
* Graph memory is *not* released automatically after ``backward`` (batching
  multiple losses is a common pattern); call :meth:`Tensor.free_graph` once a
  graph is finished with.
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import math
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Tensor",
    "TensorStore",
    "Function",
    "Context",
    "no_grad",
    "is_grad_enabled",
    "set_grad_enabled",
]

DEFAULT_DTYPE = np.float32


# ---------------------------------------------------------------------------
# Tensor store
# ---------------------------------------------------------------------------
class TensorStore:
    """Process-local arena holding all tensor buffers.

    Tensors built via :meth:`new` reference the store by index; the wrapper
    object itself stores only integers.  This is the reason a graph can be
    created and destroyed millions of times per training step with very little
    Python object overhead.
    """

    __slots__ = ("arrays", "grads", "refs")

    def __init__(self) -> None:
        self.arrays: list[np.ndarray] = []
        self.grads: list[Optional[np.ndarray]] = []
        # Number of live Tensors referencing a slot.  When it drops to zero on
        # a graph node, the slot may be reclaimed.
        self.refs: list[int] = []

    # -- allocation --------------------------------------------------------
    def new(self, data: Any, requires_grad: bool = False) -> int:
        """Register a buffer.  Gradient storage is allocated lazily.

        Graph outputs are created hundreds of times per step; pre-allocating a
        zeroed gradient buffer for each of them dominated allocation time.  A
        slot only gets a gradient array when something actually accumulates
        into it, which matches the ``None``-until-used convention of the
        ``.grad`` attribute in most autograd libraries.
        """
        arr = np.asarray(data, dtype=DEFAULT_DTYPE)
        idx = len(self.arrays)
        self.arrays.append(arr)
        self.grads.append(None)
        self.refs.append(1)
        return idx

    def new_output(self, arr: np.ndarray) -> int:
        """Register an operation result (never needs a gradient buffer yet)."""
        idx = len(self.arrays)
        self.arrays.append(arr)
        self.grads.append(None)
        self.refs.append(1)
        return idx

    def new_empty_like(self, idx: int) -> int:
        arr = self.arrays[idx]
        new = np.empty_like(arr)
        out = len(self.arrays)
        self.arrays.append(new)
        self.grads.append(None)
        self.refs.append(1)
        return out

    # -- data access -------------------------------------------------------
    def data(self, idx: int) -> np.ndarray:
        return self.arrays[idx]

    def grad(self, idx: int) -> np.ndarray:
        g = self.grads[idx]
        if g is None:
            g = np.zeros_like(self.arrays[idx])
            self.grads[idx] = g
        return g

    def accumulate_grad(self, idx: int, g: np.ndarray) -> None:
        cur = self.grads[idx]
        if cur is None:
            # Take ownership when the buffer is freshly computed, copy otherwise
            # (the same incoming array may be accumulated into several slots).
            self.grads[idx] = np.asarray(g, dtype=DEFAULT_DTYPE)
        else:
            np.add(cur, g, out=cur)

    def set_data(self, idx: int, arr: np.ndarray) -> None:
        self.arrays[idx] = np.asarray(arr, dtype=DEFAULT_DTYPE)

    def release(self, idx: int) -> None:
        """Drop references to a slot (called when a Tensor is garbage-collected).

        The buffer is replaced by a shared zero-length array (no allocation) so
        the real array can be freed immediately instead of waiting for the slot
        list to be rebuilt.
        """
        if idx < 0 or idx >= len(self.arrays):
            return
        self.refs[idx] -= 1
        if self.refs[idx] <= 0:
            self.arrays[idx] = _EMPTY
            self.grads[idx] = None

    # -- serialisation -----------------------------------------------------
    def clone_dense(self) -> "TensorStore":
        """Copy referenced slots into a fresh store (for pickled tensors)."""
        new = TensorStore()
        new.arrays = [np.array(a, dtype=DEFAULT_DTYPE, copy=True) for a in self.arrays]
        new.grads = [None if g is None else np.array(g, dtype=DEFAULT_DTYPE, copy=True) for g in self.grads]
        new.refs = list(self.refs)
        return new


_EMPTY = np.zeros((0,), dtype=DEFAULT_DTYPE)

_STORE: contextvars.ContextVar[TensorStore] = contextvars.ContextVar("boxz_store")


def store() -> TensorStore:
    """Return the active tensor store, creating it lazily."""
    try:
        return _STORE.get()
    except LookupError:
        st = TensorStore()
        _STORE.set(st)
        return st


def set_store(st: TensorStore) -> contextvars.Token:
    return _STORE.set(st)


def reset_store(token: contextvars.Token) -> None:
    _STORE.reset(token)


def current_store_or_none() -> Optional[TensorStore]:
    try:
        return _STORE.get()
    except LookupError:
        return None


# ---------------------------------------------------------------------------
# Gradient modes
# ---------------------------------------------------------------------------
_GRAD_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar("boxz_grad_enabled", default=True)


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED.get()


@contextlib.contextmanager
def set_grad_enabled(mode: bool) -> Iterator[None]:
    tok = _GRAD_ENABLED.set(mode)
    try:
        yield
    finally:
        _GRAD_ENABLED.reset(tok)


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph construction (inference mode)."""
    tok = _GRAD_ENABLED.set(False)
    try:
        yield
    finally:
        _GRAD_ENABLED.reset(tok)


# ---------------------------------------------------------------------------
# Functions, contexts and the graph node
# ---------------------------------------------------------------------------
class Context:
    """Information saved by a :class:`Function` for its backward pass."""

    __slots__ = ("shapes", "needs_input_grad", "data", "args")

    def __init__(self, shapes: Tuple[Tuple[int, ...], ...], needs_input_grad: Tuple[bool, ...]) -> None:
        self.shapes = shapes
        self.needs_input_grad = needs_input_grad
        self.data: Any = None
        self.args: Optional[Tuple[Any, ...]] = None

    def save(self, **kwargs: Any) -> None:
        self.data = kwargs


class Function:
    """Base class for differentiable operations (mirrors the autograd.Function idea)."""

    name = "function"

    def forward(self, ctx: Context, *inputs: np.ndarray, **kwargs: Any) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, ctx: Context, grad_output: np.ndarray,
                 *inputs: np.ndarray) -> Tuple[Optional[Any], ...]:  # pragma: no cover
        """Return one gradient (or ``None``) per forward argument.

        ``inputs`` are the forward arguments exactly as ``forward`` received
        them, which lets an operation recover constant operands (masks, axis
        indices, epsilon) without re-deriving them.

        Unary (and last-input) buffers usually carry the parameter *saved in
        forward* rather than the operand itself, because materialising the
        inputs of every node would be memory-inefficient.

        Invariants every implementation must respect:

        1. Buffers saved in ``forward`` via ``ctx.save`` alias live tensors and
           must be treated as read-only.
        2. In-place arithmetic is only allowed on arrays freshly allocated
           inside ``backward`` (or on the gradient buffer handed in).
        3. When the elementwise speed of in-place updates is wanted, copy first:
           ``tmp = np.multiply(a, 2.0); tmp += b``.
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> "Tensor":
        return apply(self, args, kwargs)


class _Node:
    """One differentiable operation in the graph.

    ``inputs`` holds a Tensor per forward argument (constants share the
    ``_CONST`` sentinel); ``values`` keeps the *actual* argument value for those
    constants (shapes, axis indices, masks, scalars) so the backward pass can be
    replayed exactly as the forward pass saw it.
    """

    __slots__ = ("fn", "ctx", "inputs", "values")

    def __init__(self, fn: Function, ctx: Context, inputs: Tuple["Tensor", ...],
                 values: Optional[Tuple[Any, ...]] = None) -> None:
        self.fn = fn
        self.ctx = ctx
        self.inputs = inputs
        self.values = values


def apply(fn: Function, args: Sequence[Any], kwargs: Optional[dict] = None) -> "Tensor":
    """Run ``fn.forward`` and attach a graph node when gradients are enabled."""
    kwargs = kwargs or {}
    tensors: list[Tensor] = []
    np_inputs: list[Any] = []
    for a in args:
        if isinstance(a, Tensor):
            tensors.append(a)
            np_inputs.append(a.data)
        else:
            tensors.append(_CONST)
            # Non-tensor arguments (shapes, indices, scalars) are passed through
            # untouched: only float64 arrays are narrowed to the project dtype.
            if isinstance(a, np.ndarray) and a.dtype == np.float64:
                a = a.astype(DEFAULT_DTYPE)
            np_inputs.append(a)

    needs_grad = is_grad_enabled() and any(t.requires_grad for t in tensors)
    shapes = tuple(np.shape(x) for x in np_inputs)
    ctx = Context(shapes=shapes, needs_input_grad=tuple(needs_grad and t.requires_grad for t in tensors))
    with np.errstate(over="ignore", invalid="ignore"):
        out_data = fn.forward(ctx, *np_inputs, **kwargs)

    st = store()
    out = Tensor._wrap(out_data, requires_grad=needs_grad, store_=st)
    if needs_grad:
        out._node = _Node(fn, ctx, tuple(tensors), tuple(np_inputs))
    return out


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------
class Tensor:
    """A NumPy-backed tensor participating in the autograd graph.

    ``Tensor`` objects compare by identity and are cheap: the payload lives in
    a :class:`TensorStore` slot referenced by ``_idx``.
    """

    __slots__ = ("_idx", "_node", "requires_grad", "_store", "_weak")

    # -- construction ------------------------------------------------------
    def __init__(
        self,
        data: Any = None,
        requires_grad: bool = False,
        store_: Optional[TensorStore] = None,
        _idx: Optional[int] = None,
        _no_init: bool = False,
    ) -> None:
        if _no_init:
            return
        st = store_ or store()
        if isinstance(data, Tensor):
            data = data.data
        if _idx is None:
            idx = st.new(data, requires_grad=requires_grad)
        else:
            idx = _idx
        self._store = st
        self._idx = idx
        self._node: Optional[_Node] = None
        self.requires_grad = requires_grad
        self._weak = None

    @classmethod
    def _wrap(cls, data: np.ndarray, requires_grad: bool, store_: TensorStore) -> "Tensor":
        t = cls(_no_init=True)
        t._store = store_
        t._idx = store_.new(data, requires_grad=requires_grad)
        t._node = None
        t.requires_grad = requires_grad
        t._weak = None
        return t

    def __del__(self) -> None:
        try:
            if self._weak is None and self._store is not None:
                self._store.release(self._idx)
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    # -- basic properties --------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        return self._store.arrays[self._idx]

    @data.setter
    def data(self, value: Any) -> None:
        self._store.arrays[self._idx] = np.asarray(value, dtype=DEFAULT_DTYPE)

    @property
    def grad(self) -> Optional[np.ndarray]:
        return self._store.grads[self._idx]

    @grad.setter
    def grad(self, value: Any) -> None:
        self._store.grads[self._idx] = None if value is None else np.asarray(value, dtype=DEFAULT_DTYPE)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.reshape(-1)[0]) if self.data.size else float("nan")

    def tolist(self) -> Any:
        return self.data.tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype={self.data.dtype}{rg})"

    # -- graph manipulation ------------------------------------------------
    def detach(self) -> "Tensor":
        """Return a tensor sharing the same data buffer but detached from the graph."""
        t = Tensor._wrap(self.data, requires_grad=False, store_=self._store)
        return t

    def clone(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def free_graph(self) -> None:
        """Release the graph reachable from this tensor.

        Buffers are kept when the output tensor could itself be used again by a
        later ``backward``; set the attribute ``_node = None`` first if you are
        done with the output entirely.
        """
        node = self._node
        if node is None:
            return
        self._node = None
        seen: set[int] = set()
        stack = [node]
        while stack:
            nd = stack.pop()
            if id(nd) in seen:
                continue
            seen.add(id(nd))
            ctx = nd.ctx
            ctx.data = None
            ctx.shapes = ()
            ctx.needs_input_grad = ()
            nd.values = None
            for inp in nd.inputs:
                sub = inp._node
                if sub is not None and sub is not nd:
                    stack.append(sub)

    # -- backward ----------------------------------------------------------
    def backward(self, grad: Optional[Any] = None) -> None:
        """Backpropagate from this (scalar or non-scalar) tensor.

        For non-scalar tensors a gradient of ones is used, matching the
        convention of seeding a vector-Jacobian product.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad.data if isinstance(grad, Tensor) else grad, dtype=DEFAULT_DTYPE)
            if seed.shape != self.shape:
                seed = np.broadcast_to(seed, self.shape).copy()

        st = self._store
        st.grads[self._idx] = np.array(seed, dtype=DEFAULT_DTYPE, copy=True)

        # Topological order (iterative DFS, post-order).
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[Tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            nd = node._node
            if nd is not None:
                for inp in nd.inputs:
                    if inp.requires_grad and inp._node is not None and id(inp) not in seen:
                        stack.append((inp, False))

        for t in reversed(order):
            nd = t._node
            if nd is None:
                continue
            gout = st.grads[t._idx]
            if gout is None:
                continue
            ins = nd.inputs
            n_in = len(ins)
            with np.errstate(over="ignore", invalid="ignore"):
                # Fast paths for the two most frequent node shapes avoid building
                # an argument list for every element-wise op in the graph.
                vals = nd.values
                if n_in == 2:
                    grads = nd.fn.backward(nd.ctx, gout, vals[0], vals[1])
                elif n_in == 1:
                    grads = nd.fn.backward(nd.ctx, gout, vals[0])
                else:
                    grads = nd.fn.backward(nd.ctx, gout, *vals)
            for inp, g in zip(ins, grads):
                if g is None or not inp.requires_grad:
                    continue
                g = _coerce_grad(g)
                ist = inp._store
                if g.shape != inp.data.shape:
                    g = _unbroadcast(g, inp.data.shape)
                ist.accumulate_grad(inp._idx, g)

    # -- pickling ----------------------------------------------------------
    def __reduce__(self):
        st = self._store.clone_dense()
        return (_rebuild_tensor, (st, self._idx, self.requires_grad))

    # -- operators ---------------------------------------------------------
    def __add__(self, other: Any) -> "Tensor":
        return add(self, other)

    def __radd__(self, other: Any) -> "Tensor":
        return add(other, self)

    def __sub__(self, other: Any) -> "Tensor":
        return sub(self, other)

    def __rsub__(self, other: Any) -> "Tensor":
        return sub(other, self)

    def __mul__(self, other: Any) -> "Tensor":
        return mul(self, other)

    def __rmul__(self, other: Any) -> "Tensor":
        return mul(other, self)

    def __truediv__(self, other: Any) -> "Tensor":
        return div(self, other)

    def __rtruediv__(self, other: Any) -> "Tensor":
        return div(other, self)

    def __neg__(self) -> "Tensor":
        return neg(self)

    def __matmul__(self, other: Any) -> "Tensor":
        return matmul(self, other)

    def __rmatmul__(self, other: Any) -> "Tensor":
        return matmul(other, self)

    def __pow__(self, other: Any) -> "Tensor":
        return pow(self, other)

    def __getitem__(self, key: Any) -> "Tensor":
        return getitem(self, key)

    def __setitem__(self, key: Any, value: Any) -> None:
        setitem_(self, key, value)

    # convenience reshaping operators -------------------------------------
    @property
    def T(self) -> "Tensor":
        return transpose(self, None)

    def transpose(self, axes: Optional[Sequence[int]] = None) -> "Tensor":
        return transpose(self, axes)

    def reshape(self, *shape: Any) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def view(self, *shape: Any) -> "Tensor":
        return self.reshape(*shape)

    def flatten(self, start_dim: int = 0) -> "Tensor":
        shape = self.shape
        new = shape[:start_dim] + (int(np.prod(shape[start_dim:])),) if start_dim < len(shape) else shape
        return reshape(self, new)

    def squeeze(self, axis: Any = None) -> "Tensor":
        return squeeze(self, axis)

    def unsqueeze(self, axis: int) -> "Tensor":
        return reshape(self, self.shape[:axis] + (1,) + self.shape[axis:])

    def astype(self, dtype: Any) -> "Tensor":
        return cast(self, dtype)

    def sum(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        return sum_(self, axis, keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        return mean(self, axis, keepdims)

    def exp(self) -> "Tensor":
        return exp(self)

    def log(self) -> "Tensor":
        return log(self)

    def sqrt(self) -> "Tensor":
        return sqrt(self)

    def clip(self, min_value: Optional[float] = None, max_value: Optional[float] = None) -> "Tensor":
        return clip(self, min_value, max_value)

    def __float__(self) -> float:
        return self.item()


# Sentinel used in graph nodes for non-tensor operands.  It owns a single
# zero-length buffer so that ``node.inputs[i].data`` is always well defined.
_CONST_STORE = TensorStore()
_CONST = Tensor.__new__(Tensor)
_CONST._store = _CONST_STORE
_CONST._idx = _CONST_STORE.new(np.zeros((0,), dtype=DEFAULT_DTYPE))
_CONST._node = None
_CONST.requires_grad = False
_CONST._weak = True


def _rebuild_tensor(st: TensorStore, idx: int, requires_grad: bool) -> Tensor:
    t = Tensor._wrap(st.arrays[idx], requires_grad=requires_grad, store_=st)
    return t


def _coerce_grad(g: Any) -> np.ndarray:
    if isinstance(g, Tensor):
        g = g.data
    return np.ascontiguousarray(g, dtype=DEFAULT_DTYPE)


def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if g.shape == shape:
        return g
    # Sum over leading axes that were broadcast.
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _elementwise_dtype(*args: Any) -> np.dtype:
    return DEFAULT_DTYPE


# ===========================================================================
# Elementary operations
# ===========================================================================
class Add(Function):
    name = "add"

    def forward(self, ctx, a, b):  # type: ignore[override]
        return a + b

    def backward(self, ctx, g, a, b):  # type: ignore[override]
        return g, g


class Sub(Function):
    name = "sub"

    def forward(self, ctx, a, b):  # type: ignore[override]
        return a - b

    def backward(self, ctx, g, a, b):  # type: ignore[override]
        return g, -g


class Mul(Function):
    name = "mul"

    def forward(self, ctx, a, b):  # type: ignore[override]
        ctx.save(a=a, b=b)
        return a * b

    def backward(self, ctx, g, a, b):  # type: ignore[override]
        return g * ctx.data["b"], g * ctx.data["a"]


class Div(Function):
    name = "div"

    def forward(self, ctx, a, b):  # type: ignore[override]
        ctx.save(a=a, b=b)
        return a / b

    def backward(self, ctx, g, a, b):  # type: ignore[override]
        a_, b_ = ctx.data["a"], ctx.data["b"]
        return g / b_, -g * a_ / (b_ * b_)


class Neg(Function):
    name = "neg"

    def forward(self, ctx, a):  # type: ignore[override]
        return -a

    def backward(self, ctx, g, a):  # type: ignore[override]
        return (-g,)


class Pow(Function):
    name = "pow"

    def forward(self, ctx, a, b):  # type: ignore[override]
        ctx.save(a=a, b=b)
        return np.power(a, b)

    def backward(self, ctx, g, a, b):  # type: ignore[override]
        a_, b_ = ctx.data["a"], ctx.data["b"]
        ga = g * b_ * np.power(a_, b_ - 1) if np.isscalar(b_) or b_.ndim == 0 else g * b_ * np.power(a_, b_ - 1)
        gb = g * np.power(a_, b_) * np.log(np.where(a_ > 0, a_, 1.0)) if np.isscalar(b_) and float(b_) == 0 else None
        return ga, gb


class MatMul(Function):
    name = "matmul"

    def forward(self, ctx, a, b):  # type: ignore[override]
        ctx.save(a=a, b=b)
        return a @ b

    def backward(self, ctx, g, a, b):  # type: ignore[override]
        a_, b_ = ctx.data["a"], ctx.data["b"]
        if a_.ndim == 2 and b_.ndim == 2:
            ga = g @ b_.T
            gb = a_.T @ g
        elif a_.ndim >= 3 or b_.ndim >= 3:
            ga = np.matmul(g, _swap_last2(b_))
            gb = np.matmul(_swap_last2(a_), g)
        elif a_.ndim == 1 and b_.ndim == 2:
            ga = g @ b_.T
            gb = np.outer(a_, g)
        elif a_.ndim == 2 and b_.ndim == 1:
            ga = np.outer(g, b_)
            gb = a_.T @ g
        else:  # 1-D @ 1-D -> scalar
            ga = g * b_
            gb = g * a_
        return ga, gb


def _swap_last2(x: np.ndarray) -> np.ndarray:
    return np.swapaxes(x, -1, -2)


class GetItem(Function):
    name = "getitem"

    def forward(self, ctx, a, key):  # type: ignore[override]
        ctx.save(shape=a.shape, key=key)
        return a[key]

    def backward(self, ctx, g, a, key):  # type: ignore[override]
        # Scatter-add the incoming gradient back to the indexed positions.
        #
        # ``np.add.at`` is unbuffered and slow.  Every integer-array indexing
        # pattern the model produces (embedding lookups, shape-parameterised
        # slices like x[:, :, 0]) has a *regular* index shape, so the gradient
        # can be accumulated with a single bincount over flattened indices:
        #     slot = sum_k idx_k * stride_k,   out.ravel()[slot] += g.ravel()
        # This is exactly equivalent to np.add.at and typically 5-10x faster.
        shape = ctx.data["shape"]
        out = np.zeros(shape, dtype=DEFAULT_DTYPE)
        keys = key if isinstance(key, tuple) else (key,)
        if not any(isinstance(k, np.ndarray) for k in keys):
            # No array operand: nothing can repeat, so the scatter is exact.
            # Integer scalars are re-applied as slices (1:2) so that NumPy does
            # not collapse a dimension and change the shape of the target view.
            scalar_key = tuple(slice(k, k + 1) if isinstance(k, (int, np.integer)) and not isinstance(k, bool)
                               else k for k in keys)
            out[scalar_key] += np.reshape(g, out[scalar_key].shape)
            return out, None
        if any(isinstance(k, (slice, type(Ellipsis))) or k is None for k in keys):
            np.add.at(out, key, g)
            return out, None
        other = out
        for k in keys[:-1]:
            other = other[k]
        idx = keys[-1]
        if not (isinstance(idx, np.ndarray) and idx.dtype.kind in "iu"):
            np.add.at(out, key, g)
            return out, None
        if idx.size and (idx.min() < 0 or idx.max() >= other.shape[0]):
            np.add.at(out, key, g)
            return out, None
        # Whole-row indexing: out[idx] has shape idx.shape + row_shape, so the
        # incoming gradient is g.reshape(idx.size, row_size).
        row_size = int(np.prod(other.shape[1:], dtype=np.int64)) if other.ndim > 1 else 1
        rows = int(idx.size)
        view = out.reshape(shape[0], -1) if other is out else None
        if view is not None and row_size == view.shape[1]:
            # Rows are contiguous in the flat buffer: one add_along_axis and a
            # small ufunc.at per row is ~4x faster than a bincount here.
            g2 = np.ascontiguousarray(g, dtype=DEFAULT_DTYPE).reshape(rows, row_size)
            if rows == 1:
                np.add(out[idx.reshape(-1)[0]], g2[0], out=out[idx.reshape(-1)[0]])
                return out, None
            flat = idx.reshape(-1).astype(np.int64)
            order = np.argsort(flat, kind="stable")
            sorted_idx = flat[order]
            starts = np.flatnonzero(np.r_[True, sorted_idx[1:] != sorted_idx[:-1]])
            sums = np.add.reduceat(g2[order], starts, axis=0, dtype=np.float64)
            np.add.at(view, sorted_idx[starts], sums.astype(DEFAULT_DTYPE))
            return out, None
        g2 = np.asarray(g, dtype=np.float64).reshape(rows, row_size)
        base = idx.reshape(rows, 1).astype(np.int64) * row_size + np.arange(row_size, dtype=np.int64)
        acc = np.bincount(base.reshape(-1), weights=g2.reshape(-1),
                          minlength=other.shape[0] * row_size)
        other[...] = acc.reshape(other.shape[0], row_size).astype(DEFAULT_DTYPE).reshape(other.shape)
        return out, None


class SetItem(Function):
    name = "setitem"

    def forward(self, ctx, a, key, value):  # type: ignore[override]
        ctx.save(shape=a.shape, key=key)
        out = np.array(a, copy=True)
        out[key] = value
        return out

    def backward(self, ctx, g, a, key, value):  # type: ignore[override]
        ga = np.array(g, copy=True)
        ga[ctx.data["key"]] = 0
        gv = g[ctx.data["key"]]
        return ga, None, gv


class Reshape(Function):
    name = "reshape"

    def forward(self, ctx, a, shape):  # type: ignore[override]
        ctx.save(shape=a.shape)
        return a.reshape(shape)

    def backward(self, ctx, g, a, shape):  # type: ignore[override]
        return g.reshape(ctx.data["shape"]), None


class Transpose(Function):
    name = "transpose"

    def forward(self, ctx, a, axes):  # type: ignore[override]
        ctx.save(axes=axes)
        return np.transpose(a, axes)

    def backward(self, ctx, g, a, axes):  # type: ignore[override]
        axes = ctx.data["axes"]
        if axes is None:
            inv = None
        else:
            inv = tuple(np.argsort(axes))
        return np.transpose(g, inv), None


class Squeeze(Function):
    name = "squeeze"

    def forward(self, ctx, a, axis):  # type: ignore[override]
        ctx.save(shape=a.shape)
        return np.squeeze(a, axis=axis)

    def backward(self, ctx, g, a, axis):  # type: ignore[override]
        return g.reshape(ctx.data["shape"]), None


class Sum(Function):
    name = "sum"

    def forward(self, ctx, a, axis, keepdims):  # type: ignore[override]
        ctx.save(shape=a.shape, axis=axis, keepdims=keepdims)
        return np.sum(a, axis=axis, keepdims=keepdims)

    def backward(self, ctx, g, a, axis, keepdims):  # type: ignore[override]
        shape = ctx.data["shape"]
        axis = ctx.data["axis"]
        if axis is not None and not ctx.data["keepdims"]:
            g = np.expand_dims(g, axis)
        return np.broadcast_to(g, shape).copy(), None, None


class Mean(Function):
    name = "mean"

    def forward(self, ctx, a, axis, keepdims):  # type: ignore[override]
        ctx.save(shape=a.shape, axis=axis, keepdims=keepdims)
        return np.mean(a, axis=axis, keepdims=keepdims)

    def backward(self, ctx, g, a, axis, keepdims):  # type: ignore[override]
        shape = ctx.data["shape"]
        axis = ctx.data["axis"]
        if axis is None:
            n = int(np.prod(shape))
            return np.broadcast_to(g / n, shape).copy(), None, None
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        n = int(np.prod([shape[ax] for ax in axes]))
        if not ctx.data["keepdims"]:
            g = np.expand_dims(g, axes)
        return np.broadcast_to(g / n, shape).copy(), None, None


class Exp(Function):
    name = "exp"

    def forward(self, ctx, a):  # type: ignore[override]
        out = np.exp(a)
        ctx.save(out=out)
        return out

    def backward(self, ctx, g, a):  # type: ignore[override]
        return (g * ctx.data["out"],)


class Log(Function):
    name = "log"

    def forward(self, ctx, a):  # type: ignore[override]
        ctx.save(a=a)
        return np.log(a)

    def backward(self, ctx, g, a):  # type: ignore[override]
        return (g / ctx.data["a"],)


class Sqrt(Function):
    name = "sqrt"

    def forward(self, ctx, a):  # type: ignore[override]
        out = np.sqrt(a)
        ctx.save(out=out)
        return out

    def backward(self, ctx, g, a):  # type: ignore[override]
        return (g * 0.5 / ctx.data["out"],)


class Clip(Function):
    name = "clip"

    def forward(self, ctx, a, lo, hi):  # type: ignore[override]
        ctx.save(a=a, lo=lo, hi=hi)
        return np.clip(a, lo, hi)

    def backward(self, ctx, g, a, lo, hi):  # type: ignore[override]
        a_, lo_, hi_ = ctx.data["a"], ctx.data["lo"], ctx.data["hi"]
        mask = np.ones_like(a_, dtype=bool)
        if lo_ is not None:
            mask &= a_ > lo_
        if hi_ is not None:
            mask &= a_ < hi_
        return g * mask, None, None


class Cast(Function):
    name = "cast"

    def forward(self, ctx, a, dtype):  # type: ignore[override]
        return a

    def backward(self, ctx, g, a, dtype):  # type: ignore[override]
        return g, None


# thin public wrappers -------------------------------------------------------
add = Add()
sub = Sub()
mul = Mul()
div = Div()
neg = Neg()
pow = Pow()
matmul = MatMul()
getitem = GetItem()
setitem_ = SetItem()
reshape = Reshape()


class _TransposeDefault(Function):
    """Transpose with an optional ``axes``; ``TypeError`` from the strict
    signature is avoided by normalising ``None`` inside the wrapper."""

    name = "transpose"

    def forward(self, ctx, a, axes=None):  # type: ignore[override]
        ctx.save(axes=axes)
        return np.transpose(a, axes)

    def backward(self, ctx, g, a, axes=None):  # type: ignore[override]
        axes = ctx.data["axes"]
        inv = None if axes is None else tuple(np.argsort(axes))
        return np.transpose(g, inv), None


transpose = _TransposeDefault()
squeeze = Squeeze()
sum_ = Sum()
mean = Mean()
exp = Exp()
log = Log()
sqrt = Sqrt()
clip = Clip()
cast = Cast()


def tensor(data: Any, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])  # type: ignore[assignment]
    return Tensor(np.zeros(shape, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])  # type: ignore[assignment]
    return Tensor(np.ones(shape, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = False, std: float = 1.0, seed: Optional[int] = None) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])  # type: ignore[assignment]
    rng = np.random.default_rng(seed)
    return Tensor((rng.standard_normal(shape) * std).astype(DEFAULT_DTYPE), requires_grad=requires_grad)


def arange(*args: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.arange(*args, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)


def broadcast_to(t: Tensor, shape: Sequence[int]) -> Tensor:
    class _B(Function):
        def forward(self, ctx, a, shp):
            return np.broadcast_to(a, shp)

        def backward(self, ctx, g, a, shp):
            return g, None

    return _B()(t, tuple(shape))

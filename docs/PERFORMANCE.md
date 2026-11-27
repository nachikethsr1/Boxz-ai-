# Performance

All numbers below were measured in this repository on the reference machine:

* Intel Core i5-1135G7, **4 cores / 4 threads**, 8 GB RAM, no discrete GPU used
* Python 3.11, NumPy 2.4 with OpenBLAS 0.3.31 (`Haswell`, DYNAMIC_ARCH)

Reproduce with `python -m boxz benchmark --json benchmarks/latest.json`.

## Where the time goes

For the 3.1 M-parameter model (`d_model=192, 6 layers, ctx=128, batch=24`):

| Stage | Time | Notes |
|---|---|---|
| forward (graph building) | ≈540 ms | ~230 GFLOP of useful work |
| backward | ≈1070 ms | ~2× forward cost, as expected |
| AdamW step | <1 ms | 76 tensors, ~12 MB |
| **total** | **≈1.7 s** | ≈1 800 tokens/s |

Matmuls dominate and are already running in OpenBLAS at 175–235 GFLOP/s on the
model's shapes, which is ~85 % of the AVX2 FMA peak for this CPU. That is why
the large GEMMs are **not** rewritten in Rust: doing so would be slower.

## Optimisations that mattered (each measured)

| Change | Before → after | Why |
|---|---|---|
| `np.power(x, 3)` → `x*x*x` in GELU | 0.94 s → 0.08 s per step (GELU part) | `np.power` with an integer exponent goes through `pow`, ~12× slower than multiply |
| Avoid subnormal `exp` in attention | masked softmax 29 ms → 12 ms | masking with `-1e30` *before* softmax makes `exp` return subnormals, which are pathologically slow on x86; masking to `-30` *after* the row-max subtraction keeps results normal |
| `np.add.at` → `bincount`/`reduceat` for embedding grads | 0.13 s → 0.01 s | `np.add.at` is an unbuffered scalar loop |
| Lazy gradient allocation | −0.04 s/step | pre-allocating a zeroed buffer per graph node dominated allocation |
| Direct backward dispatch (no `*args` list) | −0.30 s/step | 196 nodes/step × list construction |
| `Tensor.free_graph()` at the end of a step | memory flat | without it each step retained ~200 intermediate buffers (~1.4 GB/step of growth until GC) |
| Fused QKV projection | 3 matmuls → 1 per layer | fewer BLAS launches, same FLOPs |
| Cache causal mask per length | mask rebuilt 6×/step → 0 | `np.tril` on 128² per layer |

## Optimisations that did **not** help (and were reverted)

Honesty matters more than a longer changelog:

* **Transposed-GEMM backward** (`g @ b.T` rewritten as `(b @ g.T).T` to let BLAS
  use a transpose flag): 4.47 ms vs 4.48 ms — no difference with this OpenBLAS
  build, reverted.
* **`np.einsum`** for odd-shaped GEMMs: 67 ms vs 4.5 ms (and still 5.6 ms with
  `optimize=True`) — `@` is always the right call.
* **Thread count**: `OPENBLAS_NUM_THREADS=1` halves throughput (92 vs 175
  GFLOP/s); 2 vs 4 threads differ by <1 %. The default is correct.
* **Full in-place GELU rewrite**: removed more time in arithmetic than it added
  in bookkeeping, but during development it introduced an aliasing bug; the
  simpler version was kept because it is equally fast on this workload and
  easier to verify.

## Inference

| Path | 100 new tokens (3.1 M model, ctx 128) |
|---|---|
| full forward each step | ≈4.2 s |
| incremental KV cache | ≈0.55 s |

The KV cache is verified to produce identical logits (max diff 6e-8) and
identical greedy output; `test_kv_cache_is_faster` guards the speedup.

## Scaling expectations

Time per step scales roughly as

```
t ≈ (12 · L · d² · B · T  +  2 · B · T · d · V) / throughput
```

with `throughput ≈ 180 GFLOP/s` on this machine. Doubling `d_model` quadruples
the matmul cost per token; doubling the batch is close to linear. Practical
guidance for this hardware is a 3–8 M-parameter model at 2 000–3 000 tokens/s.

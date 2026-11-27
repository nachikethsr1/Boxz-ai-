# Rust performance kernels

Optional CPU kernels for the Boxz engine. **Boxz runs fully without this
crate**; the NumPy implementation is the reference and the automatic fallback.

## What is accelerated, and why *only* this

The engine was profiled before any Rust was written (see
[`../../docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md) and
[`../../benchmarks`](../../benchmarks)). Findings on the reference hardware
(Intel i5-1135G7, 4 cores, NumPy/OpenBLAS):

| Operation | Share of a training step | Decision |
|---|---|---|
| Matmuls (attention, MLP, logits) | ~65–70 % | **Stay in NumPy.** OpenBLAS already reaches ~175–235 GFLOP/s here (>80 % of the AVX2 FMA ceiling). A hand-written Rust GEMM would need BLIS-grade blocking to compete; rewriting it would be an engineering loss, and the project rule is *benchmark before replacing*. |
| GELU | ~8 % | **Rust**: memory-bound elementwise pass, fused. |
| Softmax / LayerNorm | ~10 % | **Rust**: fused, single pass instead of 5–6 temporaries. |
| Sampling (per generated token) | long tail during inference | **Rust**: replaces ~10 tiny NumPy calls per token with one call. |
| Embedding gradient scatter | was ~8 % | Fixed in Python first (`bincount`/`reduceat` instead of `np.add.at`), no Rust needed. |

## Build

```bash
pip install maturin
cd rust/boxz_rust
maturin develop --release
```

or with plain `cargo`:

```bash
cargo build --release
export BOXZ_RUST_LIB=$PWD/target/release/libboxz_rust.so
```

Verify:

```bash
python -c "from boxz.backends import get_backend; print(get_backend('auto').name)"
python -m pytest tests/test_backend_parity.py -v
python -m boxz benchmark --json benchmarks/rust.json
```

## Correctness contract

* `tests/test_backend_parity.py` compares every kernel with the NumPy reference
  on random inputs (`rtol=1e-5, atol=1e-6`) and checks that both samplers
  restrict support to the same top-k/top-p sets.
* The training path never depends on Rust: `TrainConfig.backend` defaults to
  `"numpy"`, and `"rust"` degrades to NumPy with an explanatory warning.

## Status in this repository

The crate is written against `pyo3 0.22` / `rust-numpy 0.22`. The environment in
which this milestone was prepared had **no Rust toolchain and no network access
to `static.rust-lang.org`**, so it could not be compiled here. It is therefore
marked **experimental**, is excluded from the critical path, and the Python side
(loader, fallback, parity harness, benchmarks) ships and is tested against the
NumPy backend.

"""Micro-benchmarks for the numerical engine.

    python -m boxz benchmark
    python -m boxz.bench.bench --config configs/boxz-1m.json

Two kinds of measurements:

* **op micro-benchmarks** -- isolated kernels (matmul, GELU, softmax, LayerNorm,
  embedding scatter, sampling) so regressions are attributed to a specific
  operation rather than to "training got slower".
* **step benchmarks** -- forward/backward cost of a whole model configuration,
  which is what actually decides how much training fits in a session.

Results are printed as a table and can be written to JSON with ``--json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = ["bench", "run_benchmarks", "op_benchmarks", "step_benchmark"]


@dataclass
class BenchResult:
    name: str
    shape: str
    milliseconds: float
    gflops: Optional[float] = None
    note: str = ""


def bench(name: str, fn: Callable[[], object], shape: str = "", flops: Optional[float] = None,
          warmup: int = 2, repeats: int = 5, note: str = "") -> BenchResult:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt = (time.perf_counter() - t0) / repeats
    gflops = (flops / dt / 1e9) if flops and dt > 0 else None
    return BenchResult(name=name, shape=shape, milliseconds=dt * 1000, gflops=gflops, note=note)


def op_benchmarks() -> List[BenchResult]:
    from boxz.backends import get_backend
    from boxz.inference.sampler import SamplingParams, sample_token

    rng = np.random.default_rng(0)
    out: List[BenchResult] = []

    # -- matmuls (the floor of everything) ------------------------------
    for (m, k, n) in [(3072, 192, 768), (3072, 192, 2048), (3072, 768, 192)]:
        a = rng.standard_normal((m, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        out.append(bench(f"matmul {m}x{k}x{n}", lambda a=a, b=b: a @ b,
                         shape=f"({m},{k})@({k},{n})", flops=2 * m * k * n))

    # -- elementwise ----------------------------------------------------
    x = rng.standard_normal((24 * 128, 768), dtype=np.float32)
    backend = get_backend("auto")
    out.append(bench(f"gelu [{backend.name}]", lambda: backend.gelu(x), shape=str(x.shape),
                     flops=x.size))
    out.append(bench("gelu numpy reference", lambda: get_backend("numpy").gelu(x), shape=str(x.shape)))

    logits = rng.standard_normal((24, 4096), dtype=np.float32)
    out.append(bench(f"softmax [{backend.name}]", lambda: backend.softmax(logits), shape=str(logits.shape)))
    gamma = np.ones(768, dtype=np.float32)
    beta = np.zeros(768, dtype=np.float32)
    out.append(bench(f"layernorm [{backend.name}]",
                     lambda: backend.layer_norm(x, gamma, beta, 1e-5), shape=str(x.shape)))

    # -- attention core --------------------------------------------------
    B, H, T, Dh = 16, 6, 128, 32
    q = rng.standard_normal((B, H, T, Dh), dtype=np.float32)
    out.append(bench("attention q@k^T", lambda: q @ q.transpose(0, 1, 3, 2),
                     shape=f"({B},{H},{T},{Dh})", flops=2 * B * H * T * T * Dh))

    # -- embedding scatter (the op replaced by bincount) -----------------
    ids = rng.integers(0, 4096, size=(24, 128))
    g = rng.standard_normal((24, 128, 192), dtype=np.float32)

    def add_at():
        o = np.zeros((4096, 192), dtype=np.float32)
        np.add.at(o, ids, g)

    def bincount_scatter():
        o = np.zeros((4096, 192), dtype=np.float32)
        flat = ids.reshape(-1)
        order = np.argsort(flat, kind="stable")
        sorted_ids = flat[order]
        starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
        sums = np.add.reduceat(g.reshape(-1, 192)[order], starts, axis=0, dtype=np.float64)
        np.add.at(o, sorted_ids[starts], sums.astype(np.float32))

    out.append(bench("embedding grad np.add.at", add_at, shape="(24*128)->4096 rows"))
    out.append(bench("embedding grad bincount/reduceat", bincount_scatter, shape="(24*128)->4096 rows"))

    # -- sampling --------------------------------------------------------
    row = rng.standard_normal(4096).astype(np.float32)
    params = SamplingParams(temperature=0.8, top_k=40, top_p=0.95)
    r = np.random.default_rng(0)
    out.append(bench("sample_token (numpy)", lambda: sample_token(row, params, None, r),
                     shape="vocab=4096", repeats=200))
    if backend.name == "rust":
        out.append(bench("sample_token (rust)",
                         lambda: backend.sample(row, 0.8, 40, 0.95, 1.0, [], 0),
                         shape="vocab=4096", repeats=200))
    return out


def step_benchmark(model_cfg=None, batch: int = 24, context: Optional[int] = None) -> List[BenchResult]:
    from boxz.config import ModelConfig
    from boxz.model.transformer import build_model
    from boxz.engine import no_grad

    cfg = model_cfg or ModelConfig(vocab_size=4096, n_layer=6, n_head=6, d_model=192, context_length=128)
    ctx = context or cfg.context_length
    m = build_model(cfg, seed=0)
    rng = np.random.default_rng(0)
    x = rng.integers(0, cfg.vocab_size, size=(batch, ctx))
    y = rng.integers(0, cfg.vocab_size, size=(batch, ctx))

    def step():
        _, loss = m(x, y)
        loss.backward()
        m.zero_grad()
        loss.free_graph()

    res = bench(f"train step d_model={cfg.d_model} L={cfg.n_layer}", step,
                shape=f"batch={batch} ctx={ctx} vocab={cfg.vocab_size}", warmup=1, repeats=3)
    tokens = batch * ctx
    res.note = f"{tokens / (res.milliseconds / 1000):,.0f} tok/s | {m.num_parameters():,} params"
    fwd = bench("forward only", lambda: m(x, y)[0], shape=f"batch={batch} ctx={ctx}",
                warmup=1, repeats=3, note=f"{tokens / 1.0:,.0f} tok")
    with no_grad():
        inf = bench("forward no_grad", lambda: m(x, y)[0], shape=f"batch={batch} ctx={ctx}",
                    warmup=1, repeats=3)
    return [res, fwd, inf]


def _print(results: List[BenchResult]) -> None:
    print(f"{'benchmark':44s} {'shape':26s} {'ms':>9s} {'GFLOP/s':>9s}  note")
    print("-" * 110)
    for r in results:
        g = f"{r.gflops:9.1f}" if r.gflops else f"{'-':>9s}"
        print(f"{r.name:44s} {r.shape:26s} {r.milliseconds:9.3f} {g}  {r.note}")


def run_benchmarks(config_path: Optional[str] = None, json_path: Optional[str] = None) -> List[BenchResult]:
    print(f"python {platform.python_version()} | {platform.processor() or platform.machine()}")
    results = op_benchmarks()
    print("\n== operations ==")
    _print(results)

    model_cfg = None
    if config_path:
        from boxz.config import TrainConfig

        tc = TrainConfig.load(config_path)
        model_cfg = tc.model
    steps = step_benchmark(model_cfg)
    print("\n== model steps ==")
    _print(steps)

    if json_path:
        payload = {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "ops": [asdict(r) for r in results],
            "steps": [asdict(r) for r in steps],
        }
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {json_path}")
    return results + steps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Boxz micro-benchmarks")
    p.add_argument("--config", default=None)
    p.add_argument("--json", default="benchmarks/latest.json")
    args = p.parse_args(argv)
    run_benchmarks(args.config, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

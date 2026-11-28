# BOXZ AI — PHASE 1 REPORT

**Scope:** measurement, training reliability, compute feasibility, and the
vocab/context decision. No architecture change. No new large-scale training.
No checkpoint, dataset, or template was deleted or overwritten.

**Branch:** `arena/5bdc9efb-boxz-ai` (base `adc2fcad`)
**Test environment:** Python 3.11.2, NumPy 2.4.6 / scipy-openblas 0.3.31, Linux, 2 vCPU, no GPU.

---

## 1. FILES CHANGED

### Modified (5)

| File | Change | Lines |
|---|---|---|
| `boxz/training/trainer.py` | 4 defect fixes (see §2): honour `grad_clip`, unify log schema, non-finite guard, checkpoint data-RNG | +43 / −8 |
| `boxz/training/checkpoint.py` | `load_checkpoint` now refuses registry-marked-unusable checkpoints unless `allow_unusable=True` | +14 / −1 |
| `boxz/eval/v3_suite.py` | registry gate, `--force`, timestamped sibling output instead of overwriting, `generated_at` | +24 |
| `boxz/cli.py` | refused checkpoint → one-line message on stderr, exit 2 (was an uncaught traceback) | +11 / −1 |
| `.gitignore` | `reports/neural/` is versioned; everything else under `reports/` stays transient | +6 |

### New (5)

| File | Purpose | Lines |
|---|---|---|
| `boxz/training/registry.py` | `CheckpointStatus`, `checkpoint_status`, `require_usable`, `usable_checkpoints` | 82 |
| `checkpoints/registry.json` | Per-checkpoint usability; `v4-sft/best` = unusable, `default` = `v3-sft/best` | — |
| `boxz/eval/neural_bench.py` | Neural-only benchmark: 9 probes + held-out teacher-forced accuracy, timestamped immutable output | 208 |
| `tests/test_training_reliability.py` | 18 regression tests covering every defect and the registry | 265 |
| `reports/neural/*.json`, `index.jsonl` | Versioned benchmark results | — |

**Not committed:** `scratch/` (309 KB of working evidence from Phase 0) is untracked
and not in `.gitignore`. It is intentionally left untouched — say the word if you
want it tracked or ignored.

---

## 2. BUGS FIXED

All four were found by auditing the trainer against its own declared config, and
each was **proven necessary by individual ablation** (revert one → exactly one
test group fails).

### BUG 1 — Gradient clipping never ran (critical)
`trainer.py:101` hardcoded `grad_clip=None` while `TrainConfig.grad_clip` defaulted
to `1.0` and every config file specified `1.0`. The setting was silently discarded.
**Effect:** the model trained with unbounded gradients for the entire v1–v3 lineage.
**Fix:** `AdamW(..., grad_clip=cfg.grad_clip)`.
**Proof:** pre-clip norms are now logged — `8.06, 3.60, 3.33, 3.36` — and clipped to 1.0.
**Ablation:** reverting this line alone fails `test_grad_clip_is_honoured_from_config`.

### BUG 2 — Mixed log schema (the "NaN val loss" artefact)
Train rows and validation rows had different key sets, so any consumer treating
`train_loss` as a universal column read `NaN` on every validation row.
**The NaN was an analysis artefact, not a training failure.** No non-finite loss
has ever been written by a real run.
**Fix:** unified rows — all rows carry `{kind, step, tokens_seen, elapsed_s}`;
`kind="train"` adds `train_loss/lr/grad_norm`, `kind="val"` adds `val_loss/val_ppl`.
**Ablation:** with the original writer, 8 of the 16 reliability tests fail
(`F.F.FF…`) and the run then hangs.

### BUG 3 — No non-finite-loss guard
A diverging run would silently produce NaN weights and checkpoint them.
**Fix:** `math.isfinite` check on loss and gradient norm raises `FloatingPointError`
at the offending step, before the optimizer step and before any save.
**Ablation:** reverting fails `test_non_finite_loss_stops_training`.

### BUG 4 — Data-sampling RNG never checkpointed (critical for resume)
`self.rng` fed `corpus.batch(...)` but was absent from `train_state`, so a resumed
run replayed the batch stream from the seed origin instead of continuing.
**Resume was silently not reproducible.**
**Fix:** `rng_state` written in `_save`, restored in `_resume` when present.
**Ablation:** reverting fails `test_resume_restores_data_rng` and
`test_resume_continues_the_same_batch_sequence` (2 tests).

### BUG 5 — Checkpoint guard applied at only 2 of 10 call sites
The registry gate initially covered `v3_suite` and `neural_bench`, but
`load_checkpoint` is called from `cli.py` (4×), `app/server.py`, `app/demo.py`,
`eval/compare.py` and `eval/fixed_eval.py` — all still able to load the aborted
checkpoint. Patching callers one by one was the wrong shape.
**Fix:** the guard moved into `load_checkpoint` itself, with `allow_unusable=True`
as a deliberate opt-out for forensics. Nothing is deleted; the bad checkpoint stays
on disk and stays inspectable.
**Ablation:** removing the guard fails `test_load_checkpoint_refuses_unusable_by_default`.

### Investigated, NOT bugs
- **v4-SFT checkpoint**: genuinely aborted at step 100 (val 4.6969 vs 3.0744 for v3),
  identical file size and identical `stage` label to v3. It is correctly marked
  unusable and remains on disk. This label collision is precisely why the registry
  keys on **path**, not on the checkpoint's self-reported metadata.
- **KV-cache vs no-cache divergence**: measured, then explained. See §5.

---

## 3. TESTS PASSED / FAILED (exact)

```
$ python -m pytest tests/
434 passed, 8 skipped in 12.74s
```

- **Baseline before Phase 1:** 416 passed, 8 skipped.
- **After Phase 1:** 434 passed, 8 skipped (**+18**, 0 failed, 0 errors).
- The 8 skips are pre-existing and unrelated (optional-backend tests).
- `tests/test_training_reliability.py` alone: **18 passed in 1.07 s**.

New coverage: log schema, no-NaN, grad-clip honoured / `None` passthrough, grad-norm
bounded, NaN stop, Inf stop, RNG restore, batch-sequence continuity, step/tokens
restore, seed bitwise-identity, eval determinism, registry ×4, load-guard ×2.

**Checkpoint integrity:** `sha256sum checkpoints/v3-sft/best/*` compared before and
after the entire Phase 1 — **byte-identical**.

---

## 4. TRAINING RELIABILITY RESULTS

| Property | Status | Evidence |
|---|---|---|
| Loss calculation | Pass | Finite across smoke runs; schema unified |
| Gradient calculation | **Fixed** | Clipping now fires; norms logged 8.06→3.60→3.33→3.36 |
| Optimizer update | Pass | AdamW bias-corrected; step counter restored on resume |
| Validation procedure | Pass | Deterministic across repeated runs |
| Checkpoint save/load | Pass | Round-trips; refused if registry-unusable |
| Training resume | **Fixed** | Step, tokens, and data-RNG stream all continue correctly |
| Reproducibility | Pass | Same seed → bitwise-identical weights |
| Numerical instability | **Fixed** | Non-finite loss/grad raises before corrupting state |

**Neural-only benchmark (the only admissible evidence of model ability):**

| Run | Checkpoint | Probes | Next-token acc |
|---|---|---|---|
| 1 | `v3-sft/best` | 1/9 = 0.1111 | 0.2765 |
| 2 | `v3-sft/best` | 1/9 = 0.1111 | 0.2765 |
| 3 | `v3-sft/best` | 1/9 = 0.1111 | 0.2765 |

Three independent runs, identical to four decimals — reproducible, and **no
improvement claimed**, because no retraining occurred. `v4-sft/best` is refused
with exit 2.

---

## 5. COMPUTE FEASIBILITY

### Hardware (measured, not assumed)
- **2 vCPU**, ~3 GB RAM visible, **no GPU** (no `nvidia-smi`, no `/dev/nvidia*`).
- Engine is **pure NumPy + OpenBLAS**; `cupy`/`torch`/`jax`/`numba` are all absent.

### Throughput (measured, 5-step steady state, real production config)

| Workload | Rate |
|---|---|
| Training (fwd+bwd+AdamW+clip, b=2 × 512) | **~403 tok/s** (2.54 s/step) |
| Generation, no KV cache | **24.1 tok/s** |
| Generation, KV cache | **102.3 tok/s** (4.2×) |

### Acceleration candidates — each benchmarked before being accepted or rejected

| Candidate | Measured result | Verdict |
|---|---|---|
| `OPENBLAS_NUM_THREADS=1/2/4` | 0.539 / 0.308 / 0.321 s per 300 matmuls (raw) | 4 threads is **worse** (2 cores) |
| Same, on a **full step** | 403 (default) vs 406–411 tok/s | **1–3% ≈ noise. Rejected.** |
| float64→float32 | Engine already float32 throughout | **No gain available** |
| Larger vocab | +7% params | Rejected — see §6 |

The raw-matmul speedup (1.75×) does **not** transfer to the training step (≈1.0–1.1×).
The engine is bound by Python dispatch and per-op `float32` arithmetic, not BLAS.
**Recommendation: do not tune thread counts.** Any future claim here must be
re-measured on a full step, not a microbenchmark.

### Memory (measured)
Peak RSS for batch=2 × ctx=512: **2.0 GB** (143 MB model → +978 MB forward → +892 MB backward).
That is why batch size is 2 on this machine. `free_graph()` releases the tape but
the allocator retains the arena, so RSS stays near peak.

### Training time per token count (at 403 tok/s)

| Tokens | Wall clock |
|---|---|
| 1M | 0.7 h |
| 3M (a serious SFT run) | 2.1 h |
| 10M | 6.9 h |
| 100M | 68.9 h (2.9 days) |
| 1B | 689 h (28.7 days) |
| 10B | 6,893 h (287 days) |

### Does the NumPy engine support GPU?
**No.** It is CPU-only by construction: 2,149 lines of engine/model code containing
**317 `np.` call sites (57 distinct NumPy functions)**, with hand-written backward passes and a tape-based
autograd. There is no device abstraction layer.

**What GPU support would actually require** (ordered by cost):
1. An `xp` backend indirection through all 317 call sites, **or** a full port to
   PyTorch/JAX — realistically a rewrite of `engine/` + `model/`, not a patch.
2. Reproducing bitwise-equivalence against the NumPy reference for the existing
   tests, including the new seed-identity test.
3. A GPU host. **This machine has none**, so no candidate can be validated here.
4. Only then does a 10–50× training speedup become available.

**Not recommended for Phase 2.** The cost is a multi-week rewrite whose primary
payoff — faster large runs — cannot be exercised on hardware that does not exist
in this environment. Revisit if a GPU host is provided.

### KV cache — the one speedup that is both real and already available
Measured **4.2×** (24.1 → 102.3 tok/s). I initially saw the cache and the no-cache
loop disagree, and treated it as a suspected correctness bug. It is not:
on identical prefixes the two agree at **every one of 12 steps** (max |logit diff|
0.0004–0.02, float32 accumulation), and the single observed divergence occurred at a
step where top1−top2 was **0.016** — smaller than that noise floor, i.e. a near-tie.
`_forward_one_step` correctly applies the positional offset, and the `ctx//2`
re-prime path was exercised with a 604-token prompt: 27 tokens produced, 27 returned.
**It is the production chat decoder (`ChatEngine.reply`), and it is sound** — but it
had no equivalence test. Adding one is a Phase 2 candidate.

---

## 6. VOCAB 8192 / CTX 1024 — ANALYSIS AND RECOMMENDATION

### Tokenizer coverage (measured on the shipping 4096 tokenizer)

| Text | chars/token |
|---|---|
| `game.js` | 2.22 |
| `index.html` | 2.30 |
| `style.css` | 2.23 |
| English prose (`Data.txt`) | **3.66** |

**Zero `<unk>` tokens on code (0/3,157).** Bytes are covered; the problem is not
unknown characters, it is merge efficiency on code.

### What 8192 would buy (measured — two tokenizers trained on the same 400 K sample)

| Vocab | Train time | js chars/tok | py chars/tok | corpus chars/tok |
|---|---|---|---|---|
| 4096 | 3 s | 2.13 | 1.81 | 14.72 |
| 8192 | 5 s | **2.31** | **1.93** | **16.30** |

**≈8% better compression.** Costs: **+1,835,008 params** (+7.0% of the 26.2M model;
the tied embedding table goes 7.0%→14.0% of the model) and **invalidates every
existing checkpoint**, since token ids shift wholesale.

### Context length (measured on 4,132 SFT examples)

| Percentile | Tokens |
|---|---|
| p50 | 44 |
| p90 | 81 |
| p95 | 91 |
| p99 | 108 |
| max | 512 |
| examples > 256 tokens | **4** |
| **examples > 512 (truncated)** | **0** |

**No SFT example is truncated by ctx=512.** Whole files are the ones that do not fit:
`game.js` 2,396 and `pong.js` 2,744 tokens — and neither fits 1024 either.
Doubling to 1024 would force batch 2→1 to stay inside RAM, so **tokens/step stays
at 1024** while each sequence costs more attention work (O(ctx²)). **Net: slower per
token, no new capability.** It would only matter once whole-file generation is a goal.

### The dominant constraint is data, not vocab or context

> **CORRECTED 2026-11-27 (Phase 2A prep).** The first version of this table was
> wrong. It divided *SFT* tokens (0.21 M) by parameters and reported 0.008
> tok/param and a "2,500× shortfall". SFT tokens are not the pretraining budget,
> so that was the wrong basis. It also used an extrapolated 31.1 M corpus figure.
> Both are replaced below with measured values.

| Quantity | Value | How measured |
|---|---|---|
| Model parameters | 26,208,000 | `checkpoints/v3-sft/best/train_state.json` |
| Pretrain tokens **actually trained** | 1,228,800 | `train_state.json` (1200 steps × 1024) |
| **Tokens per parameter (trained)** | **0.0469** | 1,228,800 / 26,208,000 |
| Pretrain corpus on disk | 27,681,895 tokens | tokenized with the shipping tokenizer |
| Tokens per parameter (full corpus) | 1.0562 | 27,681,895 / 26,208,000 |

**On the "compute-optimal" reference:** Chinchilla's ~20 tokens/parameter was fitted
on models of 70 M–16 B parameters. Extrapolating it down to a 26 M-parameter model
trained on web-code is **not an established result**, and this report does not
treat it as a target. What *is* solidly measured is far simpler: **0.0469 tok/param
was trained on**, i.e. the model saw less than 5% of one token per weight — barely
0.047 passes over a parameter count's worth of text. Whatever the "right" number
is, that is clearly too few, and the corpus on disk (1.06 tok/param) is ~22× more
than was used.

> **CORRECTION (Phase 2A corpus audit):** the 27.68 M figure is itself inflated.
> 98.4% of `pretrain.txt` is the same npm block written **twice**, so the unique
> corpus is **13,967,647 tokens** (0.533 tok/param). See
> `reports/PHASE2A_PLAN.md`. The conclusion below is unaffected — it is
> strengthened.

Raising vocab 4096→8192 adds 7% parameters to a model that has seen under 0.05
tokens per parameter. That makes the existing imbalance **worse**, and it buys
8% compression.

### RECOMMENDATION: **preserve vocab 4096 and ctx 512.**

Do **not** spend the next major run on a tokenizer change. Spend it on tokens: of
the 13.97 M **unique** pretraining tokens on disk, v3 consumed 1.23 M — **8.8%** —
and one full pass costs ~9.6 h at the measured rate. Vocab 8192, if ever wanted,
should follow a data investment, and it must be trained on a corpus that actually
contains the target languages (Python is currently absent from the tokenizer's
diet — see below).

### Coding-token gaps found (evidence, not speculation)
The model **refuses Python outright**:
> "I only build JavaScript, HTML and CSS projects right n…"

because the corpus contains no Python. Tokenizer evidence:
`def ` → 3 tokens, `__init__` → 6, `print(` → 4, and among JS identifiers
`console.log` → 3, `canvas` → 2, `Math.random` → 5. So the vocabulary has genuinely
learned JS idioms (`getElementById`, `addEventListener`, `querySelector` are single
tokens) but nothing beyond them. **This is a data gap that a vocab change cannot fix.**

---

## 7. RECOMMENDED PHASE 2 NEXT STEP

**`Phase 2A — Data before capacity: spend the pretrain budget already on disk.`**

Rationale: every measurement above points at the same bottleneck. v3 consumed 8.8%
of the unique pretraining tokens on disk (1.23 M of 13.97 M), both the vocab and the
context length score *below* a simple data increase on evidence-per-hour, and the
accelerations I could measure are noise (threads) or already in place (KV cache,
4.2×). GPU support is a rewrite that this hardware cannot validate.

Concretely:
1. Run a controlled pretrain on the existing corpus, keeping **vocab 4096 / ctx 512 /
   26.2M params** so results are directly comparable to v3. Sizing must respect the
   measured memory ceiling (3.0 GB peak at batch 2 × 512 against 3.94 GB total RAM),
   so this is a graded sequence, not one long run — see `reports/PHASE2A_PLAN.md`.
2. Fix the model's data diet to include Python and multi-file project structure —
   without that, no amount of capacity will produce Python or novel projects.
3. Re-benchmark with `boxz.eval.neural_bench` against the frozen v3 baseline
   (probes 1/9, next-token 0.2765). Accept only statistically clear improvement.
4. Add a KV-cache equivalence test while touching inference.
5. Leave vocab and context alone until tokens/parameter is within an order of
   magnitude of sane.

**Phase 1 is complete. Stopping here for approval before Phase 2, as instructed.**

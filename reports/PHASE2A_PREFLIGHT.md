# G1 PREFLIGHT — RESULTS

All five checks run before training started. **Two findings blocked the run and
were fixed first.**

---

## CHECK 1 — is the trainer reading `pretrain_dedup.txt`?

**⚠️ FAILED on the first attempt, then fixed.**

The config named `pretrain_dedup.txt`, but the trainer reported only
**8,293,215 tokens** for a file containing **13,967,647** — 40% of the corpus
silently vanished.

**Cause:** `DataConfig.dedupe` defaults to **`True`**, and line-level dedupe keeps
only the *first* occurrence of each line. On code that is catastrophic:

| `dedupe` | Lines that are exactly `}` | Lines that are exactly `{` |
|---|---|---|
| `True` (default) | **1** | **1** |
| `False` (v3's setting) | 53,554 | 44,145 |

With the default, the entire corpus retains **exactly one closing brace** — it
would train on syntactically broken JavaScript. v3 correctly set `dedupe: false`.

The G1 config now states `"dedupe": false` **explicitly** rather than relying on a
default, and `"exclude_sft_holdout": false, "sft_holdout": 0` to match v3.

**Verified after fix:**

| Quantity | Value |
|---|---|
| Resolved dataset | `data/processed/v3/pretrain_dedup.txt` |
| tokens_train | **12,762,728** |
| tokens_val | **671,723** |
| Total | **13,434,451** |
| vocab | 4096 |
| tokens/step | 1,024 |
| steps for G1 | 12,400 → **12,697,600 tokens** |

(The 533,196-token difference between 13,967,647 raw and 13,434,451 cleaned is
whitespace normalisation: 11.0% of characters are indentation, which
`normalise_whitespace` collapses. Verified independently — `clean_text` alone
reproduces 13,434,451 exactly.)

**✅ PASS** — the trainer reads the deduplicated file, confirmed by behaviour.

---

## CHECK 2 — warm start: weights only?

**✅ PASS**

| Property | Value |
|---|---|
| step at start | **0** |
| tokens_seen at start | **0** |
| AdamW moments | **all zero** |
| LR schedule @ step 0 | 5.0e-6 (fresh warmup) |
| Weights bitwise equal to v3 | **True** (all tensors) |
| Parameters | **26,208,000** |
| vocab / ctx | **4096 / 512** |
| grad_clip | **1.0** (applied) |

`init_from` is distinct from `resume`: resume restores step/optimizer/schedule.
Covered by `test_init_from_loads_weights_but_resets_progress`.

---

## CHECK 3 — checkpoint save and resume safety

**⚠️ ONE DEFECT FOUND AND FIXED.**

Resume verified working: ran 4 steps → `final` (step 4, tokens_seen 4,096,
`rng_state` saved, `optimizer.npz` present) → resumed → continued to step 6.

**But `save_checkpoint` wrote `model.npz` in place.** There is no
`RLIMIT_AS` (soft = hard = unlimited), so the OOM killer can `SIGKILL` mid-write
and leave a **truncated `.npz` that still looks like a checkpoint**. Saves now
write to a sibling temp file and `os.replace` into position, so an interrupted
save leaves the previous good file untouched.

Covered by `test_checkpoint_save_is_atomic`, which patches `np.savez` to write a
few bytes and then raise — proving the old bytes survive. Confirmed the test
**fails** when atomicity is removed, so it is not vacuous.

**Disk:** each training checkpoint is ~300 MB (model + 2 Adam moments).
8 rolling `step_` saves + `best` + `final` ≈ **3.0 GB**. Free disk **15 GB**. ✅

**✅ PASS after fix.** All G1 output goes to `checkpoints/boxz-26m-2a/g1/`.
`checkpoints/v3-sft/` is never a destination.

---

## CHECK 4 — neural benchmark before training

**✅ PASS** — 4th consecutive identical run.

| Run | Checkpoint | Probes | Next-token |
|---|---|---|---|
| 1 | v3-sft/best | 1/9 = 0.1111 | 0.2765 |
| 2 | v3-sft/best | 1/9 = 0.1111 | 0.2765 |
| 3 | v3-sft/best | 1/9 = 0.1111 | 0.2765 |
| **4 (pre-G1)** | v3-sft/best | **1/9 = 0.1111** | **0.2765** |

Preserved at `reports/neural/20261128T165746Z-checkpoints_v3-sft_best.json`
and appended to `reports/neural/index.jsonl`.

---

## CHECK 5 — memory limit and safe-stop

**✅ PASS with caveats**

| Guard | Status |
|---|---|
| Non-finite loss → `FloatingPointError` | ✅ active (`math.isfinite` on loss) |
| Non-finite grad norm | ✅ active |
| Checkpoint written **before** the raise | ✅ last good state survives |
| Atomic checkpoint writes | ✅ added this preflight |
| `max_minutes` wall-clock guard | configurable; set to `null` for G1 |
| **OOM handling** | ⚠️ **none — a SIGKILL is not catchable.** Mitigated by atomic saves: the worst case is losing progress since the last 1,500-step checkpoint (~1.5 h), never corruption. |

**Memory measured:** live worker **1.57 GB RSS** at step 100. Total RAM 3,939 MB,
~2.9 GB available. Headroom is real — the 3.0 GB figure I quoted earlier was a
worst-case isolated benchmark, not the trainer's steady state.

**v3 protection:** sha256 of all five files captured before
(`/tmp/g1_v3_before.txt`) and re-verified after the 3-step smoke run —
**byte-identical**. G1 only ever *reads* `checkpoints/v3-sft/best`.

---

## ACTUALS (live, step 100)

| Metric | Value |
|---|---|
| Steps completed | 100 / 12,400 |
| Tokens processed | 102,400 / 12,697,600 |
| Wall clock | 322.6 s |
| Rate | **3.23 s/step · 317 tok/s** |
| Projected total | **~11.1 h** (vs 8.8 h estimate — the estimate was optimistic) |
| train_loss | 6.4672 → 4.2127 |
| grad_norm | 5.346 / 2.213 / 2.328 — clipping at 1.0 verified active |
| val_loss | pending (first eval at step 500) |
| Peak RSS | 1.57 GB |

**Not a claim of intelligence.** Loss falling is expected on the training
distribution. Whether the model improved will be decided by the neural-only
benchmark after G1, compared against the frozen 0.1111 / 0.2765 baseline.

---

## VERDICT

| # | Check | Result |
|---|---|---|
| 1 | Reads `pretrain_dedup.txt` | ✅ after fixing `dedupe` default |
| 2 | Warm start = weights only | ✅ |
| 3 | Save/resume safe, correct directory | ✅ after adding atomic writes |
| 4 | Benchmark preserved | ✅ 1/9, 0.2765 |
| 5 | Memory + safe-stop, v3 protected | ✅ |

**All five pass. G1 started.** Tests: **436 passed, 8 skipped.** Commits
`f198f0c` (preflight fixes) and the run itself writes only to
`checkpoints/boxz-26m-2a/`.

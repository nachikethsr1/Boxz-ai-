# BOXZ PHASE 2A — PRETRAINING PLAN

**Status: PREPARED, NOT STARTED.** No long training run has begun.
This document is the approval gate you asked for: final corpus statistics,
training configuration, checkpoint plan and estimated runtime.

**Constraints honoured:** architecture unchanged · vocab 4096 unchanged ·
ctx 512 unchanged · 26.2M params unchanged · v3 and every prior checkpoint
preserved · nothing deleted or overwritten.

---

## 0. CORRECTIONS TO THE PHASE 1 REPORT

You were right to flag the arithmetic. Three numbers were wrong.

### Tokens per parameter

| Basis | Tokens | ÷ 26,208,000 | Verdict |
|---|---|---|---|
| v3 pretraining (actual) | 1,228,800 | **0.0469** | Your 0.047 is correct |
| SFT tokens | 210,052 | 0.0080 | ❌ what I mistakenly reported |
| Full corpus as I *estimated* it | ~31,100,000 | 1.1868 | ❌ extrapolated, not measured |
| Full corpus (your figure) | 31,500,000 | 1.2019 | Your 1.20 is correct *for that figure* |

The corpus figure you were working from (31.5 M) came from **my** Phase 1 report,
and it was an extrapolation from a 2 M-character sample. It never matched the real
file. See §1.

### The "2,500× compute-optimal gap" — withdrawn

That number came from 20 ÷ 0.008. Both inputs were wrong: 0.008 used the SFT basis,
and 20 is Chinchilla's fit over **70 M–16 B parameter** models. Extrapolating it to
26 M parameters on web-code is not an established result, so I am not claiming a
multiple. What is actually measured, and all I will assert:

- v3 trained on **1,228,800 tokens** against **26,208,000 parameters** = **0.0469 tokens per parameter**.
- That is **less than one token per weight**, over the whole pretraining run.
- The corpus on disk holds **~22× more** than was used.
- There is **no published compute-optimal target for a 26 M-parameter model**. I will
  not manufacture one.

---

## 1. CORPUS AUDIT — the important finding

### 1.1 The corpus on disk is 50% duplicated

`reports/PHASE1_REPORT.md` reported "31.1 M tokens available". The real file is
27,681,895 tokens — and **half of that is the same text twice**.

| Quantity | Value |
|---|---|
| `pretrain.txt` size | 67,549,703 chars |
| Tokens in the file | 27,681,895 |
| **Unique tokens** | **13,967,647** |
| Duplication | **49.5% discarded** |

**Cause, traced through the code — two bugs in the corpus builder:**

1. `build_pretrain_v3.py` has `extra_dirs=("data/raw",)` and walks
   `p.rglob("*.txt")`. That glob reaches `data/raw/npm/*.txt` (9 files,
   33,236,654 chars) and **appends the npm code into `pretrain_sft.txt`** — a
   file whose documented job is to hold instruction rows.
2. `combine_pretrain_v3.py` then writes `npm/*.txt` **+** `pretrain_sft.txt`
   into `pretrain.txt`. Since `pretrain_sft.txt` already contains the npm block,
   **the npm code lands in the file twice.**

Verified byte-exactly: the block starting `// three/three.cjs` appears **once at
offset 0** and **once at offset 1,076,393 inside the instruction section**, and
`sft[1076393:] == npm` is `True`. 66,473,308 of 67,549,703 chars (98.4%) is that
one repeated block.

### 1.2 Structural defect: an unterminated 33 MB "turn"

`build_pretrain_v3.py` joins chunks with `"\n\n"` and never emits `<|end|>`.
The first `<|end|>`-terminated segment in `pretrain.txt` is therefore
**33,236,778 characters** long: the entire npm corpus followed immediately by a
user question, with no role terminator between them. During pretraining the model
sees this as one gigantic "turn".

Every training window still slices to ctx=512, so nothing crashes — but the
chat-role framing the corpus is supposed to teach is broken at the seam.

### 1.3 Composition

| Section | Chars | Tokens | Share |
|---|---|---|---|
| npm code (`three`, `gsap`, `matter-js`, `pixi.js`, `phaser`, `lodash`, … filtered to `.js/.mjs/.cjs/.css/.html`) | 33,236,654 | — | 49.2% |
| Instruction rows (4,104 turns) | 1,076,393 | — | 1.6% |
| npm code again (the duplicate) | 33,236,654 | — | 49.2% |
| **Unique total** | **34,313,049** | **13,967,647** | 100% |

Instruction rows are only **3.1% of unique content** (1.08 M of 34.3 M chars).

### 1.4 Language coverage — what you asked me to check

| Language | Present? | Evidence |
|---|---|---|
| **JavaScript** | ✅ dominant | 48,160 `function`, 42,058 `const`, 4,446 `=>`, 9,597 `require(` |
| **CSS** | ✅ sparse | 23 `@keyframes` |
| **HTML** | ⚠️ **absent** | **0** `<canvas`, **0** `<!DOCTYPE` |
| **GLSL / shaders** | ✅ some | 3,827 `vec3` |
| **Natural-language explanation** | ⚠️ thin | 1.08 M chars of instruction rows, 3.1% of unique content |
| **Python** | ❌ **absent** | **0** `def f():` at line start, **0** `import x`, **0** `class C:`, **0** `__name__ == '__main__'` |

**Python is genuinely absent.** The `self.`/`import`/`def ` hits in raw grep are
JavaScript (`this.`/ESM `import`/method names), not Python. This is *why* the
model answers "I only build JavaScript, HTML and CSS projects right now" — the
refusal is an accurate description of its training data.

HTML being absent despite the fetch filter allowing `.html` is worth noting: the
fetched packages evidently shipped almost no `.html` files.

### 1.5 Duplicates and corruption

| Check | Result |
|---|---|
| Duplicate lines | 1,416,989 / 1,797,352 (78.8%) — mostly structural (`}` ×106,709, `{` ×88,290) |
| Duplicate documents (>200 ch) | 606 / 2,201 (27.5%) — this is the npm block duplication |
| NUL bytes | 0 |
| Control characters | 0 in first 5 M chars |
| U+FFFD replacement chars | 0 |
| Mojibake (`Ã©`, `â€™`, `ï»¿`, …) | 0 |
| `<|user|>` / `<|assistant|>` / `<|end|>` | 4,104 / 4,104 / 4,104 — balanced |

**No corruption.** The problem is duplication and missing languages, not encoding.

### 1.6 Can the data be used for training?

**Yes, with one correction applied.** Licence-wise the npm slice is licensed
source recorded in `data/sources/*.json` with a per-file manifest; instruction rows
are project-original CC0. I have **not** changed the corpus on disk — the original
`pretrain.txt` is untouched. I built a corrected file alongside it:

```
data/processed/v3/pretrain_dedup.txt   34,313,049 chars   13,967,647 tokens
  = npm block (once) + "\n\n" + instruction rows (once)
  verified: npm marker appears once, 4,104 user turns, tokenizer unchanged
```

The tokenizer is **byte-identical** to the one v3 used
(`sha256 213cde78…1ff37c`), so token ids remain compatible with every checkpoint.

### 1.7 One caveat I could not fix cheaply

25.3% of SFT **validation** questions (55/217) appear somewhere in the pretraining
text. Pretraining on SFT questions is by design in this pipeline
(`build_pretrain_v3` renders every SFT turn), so this is not new leakage introduced
now — but it does mean **SFT val loss is not a clean generalisation measure**, and I
will not use it as one. Phase 2A is judged on the neural-only benchmark, which uses
held-out probes, not on SFT val loss.

---

## 2. TRAINING CONFIGURATION

Everything architectural is frozen. Only the data file and schedule change.

| Setting | v3 (for comparison) | **Phase 2A** | Changed? |
|---|---|---|---|
| Parameters | 26,208,000 | **26,208,000** | no |
| `n_layer / n_head / d_model / d_ff` | 10 / 7 / 448 / 1792 | **same** | no |
| `vocab_size` | 4096 | **4096** | no |
| `context_length` | 512 | **512** | no |
| Tokenizer | `tokenizer.json` | **same file, same sha256** | no |
| Dataset | `pretrain.txt` (50% duplicated) | **`pretrain_dedup.txt`** | **yes** |
| Batch size | 2 | **2** (memory-bound, measured §4) | no |
| Tokens/step | 1,024 | **1,024** | no |
| LR / schedule | 6e-4 cosine → 6e-5 | **6e-4 cosine → 6e-5** | no |
| Warmup | 120 steps | **120 steps** | no |
| `grad_clip` | declared 1.0, **never applied** | **1.0, verified applied** | **fixed** |
| Start point | random init | **warm start from `v3-sft/best` weights** | **new** |
| Seed | 1337 | **1337** | no |

**Warm start, not resume.** I added `init_from` to `TrainConfig`: it loads *weights
only* from `v3-sft/best` and begins a new run at step 0 with fresh AdamW moments
and a fresh LR schedule. `resume` would have restored step 1200, the old optimizer
moments and the data stream — which would misalign the cosine schedule against
`max_steps` and make the result uninterpretable. Verified: step 0, `tokens_seen` 0,
optimizer moments zero, weights bitwise equal to v3 (`test_init_from_loads_weights_but_resets_progress`).

`v3-sft/best` is only ever **read**. Protected by: sha256 recorded before and after,
`checkpoints/registry.json` marking it usable-and-default, and the new run writing to
a **separate** directory `checkpoints/boxz-26m-2a/`.

---

## 3. CHECKPOINT PLAN

| Item | Plan |
|---|---|
| Output dir | `checkpoints/boxz-26m-2a/<run-name>/` — never `checkpoints/v3-sft/` |
| Intervals | every **1,500 steps** (~1.5 M tokens), plus `final` |
| Retained | `best` (lowest val) + `final` + rolling `step_XXXXXX` (`keep_last` as today) |
| Selection | `checkpoints/registry.json`; new run registered **usable** only after it beats v3 on the neural benchmark |
| If non-finite loss | trainer raises `FloatingPointError` and stops; last good checkpoint survives |
| If OOM / crash | `--resume <dir>` continues exactly (step, moments, data stream all restored) |
| v3 protection | untouched; sha256 verified `/tmp/ckpt_before.txt` style before and after |

**Logging per interval:** `kind`, `step`, `tokens_seen`, `train_loss`, `val_loss`,
`val_ppl`, `lr`, `grad_norm`, `elapsed_s` — plus peak RSS sampled by the wrapper.
Non-finite guard is active for both loss and grad norm.

---

## 4. estimated runtime — and why it must be graded

**Measured on this box just now** (not assumed):

| Measurement | Value |
|---|---|
| Steady-state fwd+bwd+AdamW+clip | **2.54 s/step** (403 tok/s) at batch 2 × ctx 512 |
| Peak RSS, isolated benchmark | **3.0 GB** (2233 → 3004 → 3012 → 3020 MB over 20 steps) |
| Peak RSS, real trainer run | **2.3 GB** (8 steps incl. eval + checkpoint save) |
| Machine RAM | **3,939 MB total**, ~3.4 GB available |
| CPU | 2 vCPU · no GPU |

Memory is the binding constraint: at 3.0 GB peak against 3.94 GB total, batch
cannot be raised, and a long uninterrupted run carries real OOM risk. Runtime is
therefore **staged**, each stage checkpointed and resumable:

| Stage | Epochs over 13.97 M unique | Tokens | tok/param | Steps | Time @403 tok/s |
|---|---|---|---|---|---|
| **G1** | 1 | 13,967,647 | 0.533 | 13,640 | **9.6 h** |
| **G2** | 2 | 27,935,294 | 1.066 | 27,280 | **19.3 h** |
| **G3** | 4 | 55,870,588 | 2.132 | 54,561 | **38.5 h** |
| **G4** | 8 | 111,741,176 | 4.264 | 109,122 | **77.0 h** |

For reference, v3 was **0.85 h** (1,228,800 tokens). **G1 alone is 11× v3's data.**

*Caveat on step time:* the real trainer measured ~3–4 s/step in my 8-step probe,
slower than 2.54 s, because eval and checkpoint saves are included. I therefore
expect **G1 to take 10–13 h, not exactly 9.6 h**, and will report the measured
figure rather than the estimate.

**Proposed path:** run **G1** first, evaluate, and decide on G2 with evidence in
hand. Do not commit to G3/G4 before seeing whether the first 14 M tokens move the
neural benchmark at all.

---

## 5. EVALUATION PLAN

Only the checker you approved counts as evidence.

| Tool | Why |
|---|---|
| `boxz.eval.neural_bench` | 9 neural-only probes + held-out teacher-forced accuracy. Same benchmark as Phase 0/1. |
| Baseline | `checkpoints/v3-sft/best` — **frozen**, 3 runs all identical: probes **1/9 = 0.1111**, next-token **0.2765** |
| New checkpoint | same benchmark, timestamped, appended to `reports/neural/index.jsonl` |
| Acceptance | report the actual numbers. **No claim of improvement** unless probes and next-token accuracy both move beyond run-to-run variation |

**Explicitly not evidence:** templates, the maths engine, the router, the coding
validator suite. Those are deterministic and cannot show that the neural model
learned anything. I will not present them as improvement.

**What I will not predict:** I am not going to say "14 M tokens will make Boxz
intelligent". For calibration — v3's 1.23 M tokens produced 1/9 probes and 27.7%
next-token accuracy on a 26 M-parameter model. Whether 14 M tokens moves that
measurably is an **empirical question this run is designed to answer**. It may not
move. If it does not, that is the finding, and it argues for data diversity
(Python, HTML, prose) rather than more repetitions of npm JavaScript.

---

## 6. WHAT IS READY RIGHT NOW

| Item | State |
|---|---|
| Corrected corpus `pretrain_dedup.txt` | ✅ built, verified (13,967,647 tokens) |
| `init_from` warm start | ✅ implemented + tested |
| Gradient clipping | ✅ active and verified |
| Non-finite guard | ✅ active |
| Checkpoint plan | ✅ separate dir, 1,500-step intervals |
| v3 protection | ✅ sha256 verified unchanged |
| Full test suite | ✅ **435 passed, 8 skipped** |
| Long training run | ⏸ **not started** |

**Awaiting your go-ahead to start G1.** If you prefer a different token budget,
say so and I will re-plan against it rather than run the default.

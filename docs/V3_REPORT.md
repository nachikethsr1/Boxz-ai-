# Boxz V3 — implementation report

Everything below is measured on the machine that ran it (**2 cores, 3.9 GB RAM,
no swap**). Where a number comes from a test or a checkpoint, the source is
named. Nothing is projected without being labelled as such.

---

## 1. Parameter count

| model | parameters | measured how |
|---|---|---|
| **V3 pretrained** (`configs/boxz-26m-run.json`) | **26,208,000** | built from the shipped config |
| V3 after SFT (`checkpoints/v3-sft/best`) | 26,208,000 | unchanged by SFT |
| **V3 target** (`configs/boxz-62m.json`) | **62,017,280** | built from the shipped config |
| previous release (`configs/boxz-1m.json`) | 3,087,360 | for reference |

The target config is `d_model=640, n_layer=12, n_head=10, d_ff=2560,
vocab=4096, ctx=512`. It is defined, validated and correct. It is **not
trained** — see §6.

## 2. Token counts (real tokenizer, vocab 4096)

Measured with `data/processed/v3/tokenizer.json` (byte-level BPE, 3,832 merges).

| dataset | examples | tokens |
|---|---|---|
| `maths.txt` | 900 | 29,269 |
| `coding.txt` | 2,600 | 116,787 |
| `Data.txt` | 2,000 | 148,368 |
| **V3 train split** | **4,104** | **206,634** (120,543 supervised) |
| **V3 val split** | **216** | **10,942** (6,337 supervised) |
| pretraining corpus (npm + SFT render) | — | **27,681,895** |

The tokenizer costs 11 tokens for `482736 + 918264` and 19 for
`What is 2,194,638 + 6,484,738?` — 2× better on digits than the old 2048 vocabulary,
which took 22 tokens for one 6-digit addition.

**Longest example: 134 tokens.** Nothing exceeds the 512-token context; the
oversize path in `prepare_v3` reported 0 dropped rows.

## 3. Train / validation split

Produced by `boxz/data/prepare_v3.py`:

```
5,500 parsed  ->  1,180 exact duplicate (question, answer) pairs removed
              ->  4,320 kept
              ->  4,104 train / 216 val
              ->  0 normalised questions in both splits
```

Per category: train `math 860 / coding 2,467 / general 777`; val `math 40 /
coding 133 / general 43`.

Exact duplicates removed were genuine byte-identical repeats — up to **20 copies**
of the same question and answer. Questions that share an answer but differ in
wording were **kept**, per the project rule that near-duplicate questions are not
duplicates.

## 4. Training results

### Pretraining (`configs/boxz-26m-run.json`)

12 layers? No — 10 layers, batch 2, ctx 512, lr 6e-4 cosine, 1,200 steps.

| step | train loss | val loss |
|---|---|---|
| 1 | 8.5365 | — |
| 60 | 6.4749 | — |
| 300 | 4.9071 | 4.79 |
| 600 | 4.3054 | — |
| 900 | 3.87 | — |
| **1200** | **3.8514** | **3.4421** |

`final_train_loss 3.8514`, `final_val_loss 3.4421`, `best_val_loss 3.4421`,
`tokens_seen 1,228,800`, `seconds 3652.4`, **336.4 tokens/s**, 2.1 GB peak RSS.
Val and train loss track each other, so there is no overfitting at this budget
— the model is under-trained, not over-trained.

### SFT (`boxz sft-v3`, 1,200 steps, batch 4, lr 3e-4)

| step | train | val |
|---|---|---|
| 100 | 4.74 | 4.28 |
| 400 | 4.18 | 3.51 |
| 800 | — | 3.25 |
| 1100 | 2.35 | 3.1013 |
| **1200** | **2.7145** | **3.0744** |

Best-by-val checkpoint is `checkpoints/v3-sft/best` at step 1200.

### The first SFT attempt failed, and why

Val loss reached 2.73 but the output was degenerate repetition
(`"a not not not not an not..."`). A **lower val loss with worse output** is the
signature of a data bug, not a hyper-parameter bug. It was:

> The header regex matched the bare word `user` or `assistant` anywhere it stood
> alone on a line, **including inside prose**. Result: 3,253 of 4,104 rows (79%)
> had `###` markup inside the question field, and 410 questions ended in
> fragments like `as` or `element`.

The model was being trained to answer questions no human would type. Fixed in
`prepare_v3._parse_headers`; verified 0/4104 rows now contain `###`;
locked down by 10 regression tests.

## 5. Best checkpoints

| stage | path | val loss |
|---|---|---|
| pretrained | `checkpoints/boxz-26m/boxz-26m/best` | **3.4421** |
| instruction-tuned | `checkpoints/v3-sft/best` | **3.0744** |

Both directories contain `model.npz`, `config.json`, `tokenizer.json` and
`train_state.json`, and load through `boxz.training.checkpoint.load_checkpoint`.

## 6. The 62M target — defined, measured, and blocked

Measured on this machine:

```
62,017,280 params  = 248 MB weights + 248 MB grads + 496 MB AdamW = ~1.0 GB
bare forward+backward, batch 1, ctx 512  = 3,391 MB peak
machine total                            = 3,939 MB
swap                                     = 0 MB
```

The first trainer run was **OOM-killed (exit 137)**. Three fixes were applied
and are in the tree:

1. **Micro-batch graphs freed immediately** (`trainer._accumulated_step`). Peak
   activation memory previously scaled with `grad_accum_steps`.
2. **Corpus built then released** before the first activation is allocated
   (~300 MB returned).
3. **Gradient checkpointing** (`boxz/training/gradckpt.py`) — estimated
   activation footprint for the 62M shape drops from 0.53 GB to **0.08 GB**
   (14%), with gradients verified identical to the eager path.

With checkpointing the arithmetic says 62M should now fit. It has **not** been
run end to end; the honest statement is that the blocker is understood and the
mitigation is implemented and unit-tested, not that the target is achieved.

## 7. Math Engine

`boxz/math/expr.py` (parser + exact evaluator) and `boxz/math/engine.py`
(word-problem shapes). **No `eval()`/`exec()`** — asserted structurally by AST
walk in `tests/test_math_engine.py::test_no_eval_anywhere`.

Exact via `fractions.Fraction`. A non-terminating quotient keeps both a readable
approximation and the exact fraction: `987654 / 321` → `3076.803738 (329218/107)`.

Supported: arithmetic with order of operations, negatives, decimals, fractions,
percentages (of / increase / decrease / what-percent), powers and roots (exact
when rational), averages, ratios and proportions, unit conversion (length, mass,
time, volume, temperature), geometry (rectangle, square, triangle, circle,
trapezium, cube, cuboid, cylinder), linear equations in one variable.

Errors are typed values, never silent: `division_by_zero`, `domain`,
`unknown_function`, `too_large`, `arity`, `bad_char`, …

**Acceptance: 73/73 = 100%** (`reports/v3_suite.json`). The arithmetic cases are
generated from a fixed seed and are **not** in the training data, so a correct
answer cannot be memorisation.

## 8. Coding Engine

`boxz/coding/engine.py`. Fence extraction (tagged and untagged), language
identification from content, and validators:

| language | validator | runs |
|---|---|---|
| JavaScript / TypeScript | `node --check` in a bounded subprocess | parsing only — never executed |
| HTML | tag balance, duplicate attributes, void elements | stdlib `html.parser` |
| CSS | brace balance, unclosed comments, `//` misuse | own scanner |
| Python | `compile()` | never executed |
| JSON | `json.loads` | — |

Tri-state result — `passed` / `failed` / `unavailable`. A validator that did not
run can never be reported as a pass; `test_js_reports_unavailable_without_node`
simulates a device with no Node.js and asserts exactly that.

**Acceptance: 13/13 = 100%**, with `node_available: true` on this machine.

## 9. Test results

```
253 passed, 0 failed  (pytest tests/)
```

| file | covers |
|---|---|
| `test_math_engine.py` | exact arithmetic, every error code, AST no-eval proof |
| `test_coding_engine.py` | tri-state validation, no-Node path, "never executes" |
| `test_router.py` | math/coding split, stateless topic switching |
| `test_integration.py` | real `ChatEngine`: maths turn reports `tokens_generated == 0` |
| `test_prepare_v3.py` | parser regressions incl. the real data files |
| `test_gradckpt.py` | checkpointed gradients match the reference |

## 10. Limitations — what V3 does **not** do

1. **The language model is weak.** Measured on 30 held-out V3 rows:
   **exact match 0/30, token accuracy 8.1%**. It produces topical web-development
   sentences with the right *shape* (and it terminates cleanly, and it emits
   fenced code), but the content is not correct. It is a 26M model trained on
   1.23M tokens; that is ~1 token per parameter, against the ~20 that would be
   needed.

2. **62M is not trained.** Configured, measured as OOM, mitigation implemented
   and unit-tested, not yet run end to end.

3. **Coding validation is syntax only.** Valid syntax is not working code.
   The engine never claims otherwise, and generated code is never executed.

4. **No code execution sandbox.** Deliberately out of scope for V3.

5. **Live-chat testing found three bugs that unit tests did not.** Shipping the
   server and talking to it exposed (a) a single shared conversation across all
   clients, which filled the 512-token context and degenerated replies to
   `"This."`, (b) conversational follow-ups ("now do 77 * 33") missing the
   router, and (c) verbal arithmetic ("subtract X from Y") unsupported even
   though that phrasing is in the training data. All three are fixed, with
   regression tests. The lesson is recorded because it generalises: the test
   suite exercises the engines, but only a real conversation exercises the
   *state*.

6. **The engines are separate from the model's weights.** A maths question is
   answered by Python, with the model contributing nothing. That is the design,
   not a limitation — but it means "Boxz can add" is a statement about the
   engine, never about the network.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install numpy pytest
python -m boxz.data.prepare_v3                       # 5,500 -> 4,104 / 216
python -m boxz.data.build_tokenizer_v3               # vocab 4096
python -m boxz.data.fetch_corpus_v3 --target-mb 22   # 33 MB npm source
python -m boxz.data.build_pretrain_v3
python -m boxz.data.combine_pretrain_v3              # 27.68M tokens
python -m boxz train --config configs/boxz-26m-run.json
python -m boxz sft-v3 --checkpoint checkpoints/boxz-26m/boxz-26m/best \
                      --steps 1200 --batch-size 4 --lr 3e-4
python -m boxz eval-v3 --checkpoint checkpoints/v3-sft/best
python -m boxz.app.demo --checkpoint checkpoints/v3-sft/best --port 8000
```

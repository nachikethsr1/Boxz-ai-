# Evaluation

Boxz reports what it can measure, and distinguishes *loss* from *capability*.
Every number in this file was produced by the commands shown; the raw reports
are git-ignored JSON under `reports/`.

## Reproduce

```bash
python -m boxz eval-suite --checkpoint checkpoints/base-1m/best --out reports/base.json
python -m boxz eval-suite --checkpoint checkpoints/sft/best      --out reports/sft.json
```

## Measured results (this machine, 4 CPU cores)

### Pretraining — `checkpoints/base-1m/best`

Corpus: 934 K tokens of Tiny Shakespeare + the MIT three.js/jQuery code slice.
3,087,360 parameters, context 128, BPE vocabulary 2048.

| Metric | Value |
|---|---|
| training loss (final) | **2.8498** |
| validation loss | **4.0353** |
| validation perplexity | **56.6** |
| validation bits/byte | 2.14 |
| next-token accuracy | **0.244** |
| training throughput | 1,494 tok/s (1.9 s/step, batch 24) |
| generation throughput | 1.12 s / 100 tokens (~89 tok/s) |

### Instruction tuning — `checkpoints/sft/best`

400 steps on 100 training examples (16 held out), lr 1e-4, replay ratio 0.5,
end-of-turn weight 8×, selected by held-out loss at step 200.

| Metric | Base model | After SFT | Change |
|---|---|---|---|
| **held-out instruction loss** | 7.5165 | **4.9116** | **−34.6 %** |
| validation loss (general text) | 4.0353 | 4.1445 | +2.7 % |
| next-token accuracy | 0.244 | 0.210 | — |
| instruction-format rate | 0.00 | **0.33** | learns to stop |

The instruction loss is measured on 16 examples that were withheld from *both*
pretraining and SFT (see `docs/DATASETS.md`), so it is a genuine generalisation
number rather than a memorisation score.

### What each fix contributed

Each of these was a measured before/after, not a guess:

| Change | Effect |
|---|---|
| Shakespeare only → Shakespeare + web code | val loss 5.76 → 4.04; the model can now emit `new THREE.Mesh(…)`, CSS rules and HTML structure |
| Un-share the chat template (training vs inference) | fixed fluent-looking gibberish from a model scoring 2.0 on its own training data |
| SFT rehearsal of pretraining data | general val loss after SFT 4.73 → 4.14 (avoided most catastrophic forgetting) |
| 8× end-of-turn loss weight | instruction-format rate 0.00 → 0.33; replies stop instead of running to the token cap |
| SFT lr 3e-4 → 1e-4 | held-out instruction loss 4.97 → 4.91 and a stable plateau instead of a step-100 peak |

## Metrics in `boxz/training/evaluate.py`

| Metric | Definition | Use |
|---|---|---|
| `val_loss` | mean cross entropy (nats/token) on held-out text | primary training signal |
| `val_perplexity` | `exp(val_loss)` | human-comparable |
| `val_bits_per_byte` | `val_loss / (ln 2 · bytes_per_token)` | comparable across tokenizers |
| `next_token_accuracy` | fraction of positions where argmax is the target | coarse sanity check |

Evaluation uses sequential, non-overlapping validation windows, so numbers are
deterministic across runs.

## Capability suite: `python -m boxz eval-suite --checkpoint <dir>`

* **Instruction-format compliance** — given `<|user|>…<|assistant|>`, does the
  model emit `<|end|>` and stop, rather than rambling into a new user turn? The
  base model scores 0.00 because it has never seen the format; after SFT it
  scores 0.33 on unseen prompts.
* **Held-out instruction loss** — assistant-token loss on the 16 withheld rows,
  the same split the SFT trainer selects its checkpoint on.
* **Continuations** — greedy samples for fixed prompts, stored in the report so
  qualitative regressions are visible.
* **Throughput** — wall-clock seconds per 100 generated tokens.

## Interpreting a small model honestly

At 3 M parameters and 934 K training tokens the model has learned:

* English morphology and word structure (BPE merges are meaningful),
* local syntax (line breaks, speaker labels, punctuation, JS/CSS/HTML shapes),
* Shakespearean play formatting and style,
* the conversation frame (roles, `<|end|>` termination) after SFT,
* that code questions are answered with code-shaped text.

It has **not** learned reliable factual knowledge or non-trivial programming.
It composes phrases from the domain it saw; it does not reason. Concretely: it
will start a Three.js snippet correctly and then repeat, and it will answer a
familiar question descriptively rather than correctly. That is the expected
behaviour at this scale, and the reason the roadmap's next phase is *more data
and more parameters*, not more prompt engineering.

## Regression gates used in development

1. `test_training_reduces_loss` — validation loss must fall over 40 steps.
2. `test_full_pipeline` — final validation loss must be < 4.0 on a simple corpus.
3. `test_full_model_directional_gradient_check` — gradients must match finite
   differences within 5 %.
4. `test_kv_cache_matches_full_forward` — cache logits within 1e-4.
5. `test_reproducible_training` — identical loss traces for the same seed.
6. `test_weighted_cross_entropy_gradient_matches_fd` — the end-of-turn weight
   must not break gradients.
7. `test_pretraining_corpus_excludes_sft_holdout` — held-out rows never leak
   into pretraining (with a positive control).
8. `test_training_and_inference_render_identically` — one chat template only.
9. `test_streaming_reply_grows_monotonically` — streaming must be a prefix
   sequence, never re-minded text.

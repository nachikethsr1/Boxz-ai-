# Evaluation

Boxz reports what it can measure, and distinguishes *loss* from *capability*.

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
  model emit `<|end|>` and stop, rather than rambling into a new user turn? This
  is the first behaviour that distinguishes a chat-capable checkpoint from a raw
  LM.
* **Held-out instruction loss** — assistant-token loss on the tail slice of the
  instruction set.
* **Continuations** — greedy samples for fixed prompts, stored in the report so
  qualitative regressions are visible.
* **Throughput** — wall-clock seconds per 100 generated tokens.

## Interpreting a small model honestly

At 3 M parameters the model has learned:

* English morphology and word structure (BPE merges are meaningful),
* local syntax (line breaks, speaker labels, punctuation),
* Shakespearean play formatting and style,
* the conversation frame (roles, `<|end|>` termination) after SFT.

It has **not** learned reliable factual knowledge or non-trivial programming,
because that requires orders of magnitude more data and compute. Claims in this
repository are limited to what the metrics show.

## Regression gates used in development

1. `test_training_reduces_loss` — validation loss must fall over 40 steps.
2. `test_full_pipeline` — final validation loss must be < 4.0 on a simple corpus.
3. `test_full_model_directional_gradient_check` — gradients must match finite
   differences within 5 %.
4. `test_kv_cache_matches_full_forward` — cache logits within 1e-4.
5. `test_reproducible_training` — identical loss traces for the same seed.

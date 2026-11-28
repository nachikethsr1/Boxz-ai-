# Boxz V3 — system architecture

Boxz V3 is a **small language model plus two deterministic engines**. The model
handles language; the engines handle things a 26M-parameter network cannot do
reliably (exact arithmetic, syntax validation).

```
                        ┌───────────────────────────┐
   user message  ──────▶│      boxz/router.py       │
                        │  (stateless, per message) │
                        └──────────┬────────────────┘
                                   │
                 ┌─────────────────┴──────────────────┐
                 │                                     │
        looks like maths?                      everything else
                 │                                     │
                 ▼                                     ▼
      ┌────────────────────┐              ┌────────────────────────┐
      │  boxz/math/engine  │              │  BoxzLM (transformer)  │
      │  exact + symbolic  │              │  generates a reply     │
      └─────────┬──────────┘              └───────────┬────────────┘
                │                                      │
        verified string                     fenced code blocks
                │                                      │
                │                                      ▼
                │                        ┌──────────────────────────┐
                │                        │ boxz/coding/engine.py    │
                │                        │ syntax-check, no exec    │
                │                        └───────────┬──────────────┘
                │                                    │
                └──────────────┬─────────────────────┘
                               ▼
                      reply to the user
```

## Why the model is not asked to do arithmetic

A 26M-parameter model stores 26M numbers. Exact addition of two 6-digit numbers
requires a representation of ~10^12 facts; memorisation is impossible and the
learned algorithm is unreliable. Measured on this project's earlier model, the
model produced `</a></a></a>` when asked for a multiplication table.

So `boxz/math/engine.py` computes with `fractions.Fraction` and the model never
touches the digits. `boxz/chat/engine.py::_math_reply` returns the engine's
string verbatim. The integration test asserts `tokens_generated == 0` for a maths
turn, which is the check that the model really is out of the loop.

## Why the coding engine only checks syntax

Executing generated code on a user's device is unsafe (filesystem, network,
secrets). `node --check` parses without running. The engine reports
`passed` / `failed` / `unavailable` — never `passed` for something it did not
check. A device without Node.js gets `unavailable` and the UI says so.

## State separation

* **Conversation history** lives in `ChatEngine.conversation`.
* **Tool state** lives in `boxz.router`, and it is *per message* — there is no
  "current topic" variable.

That is why a user can go maths → coding → science → maths across four
consecutive turns and each is routed on its own merits
(`tests/test_router.py::test_topic_switching_is_stateless`).

## Data flow for training

```
maths.txt   ─┐
coding.txt  ─┼─▶ boxz/data/prepare_v3.py ──▶ v3_train.jsonl / v3_val.jsonl
Data.txt    ─┘        (normalise, dedupe,       (leak-free split on
                       split by question)        normalised question key)

npm registry ─▶ boxz/data/fetch_corpus_v3.py ─┐
                                               ├─▶ pretrain.txt ─▶ BoxzLM
v3_train.jsonl ─▶ build_pretrain_v3.py ────────┘                  │
                                                                  ▼
                                              v3_train.jsonl ─▶ SFT ─▶ chat
```

## The three model sizes

| config | params | ctx | status |
|---|---|---|---|
| `configs/boxz-1m.json` | 3,087,360 | 128 | previous release |
| `configs/boxz-12m.json` | 11,338,880 | 512 | fits at batch ≤ 4 |
| `configs/boxz-26m-run.json` | 26,208,000 | 512 | **trained for V3** |
| `configs/boxz-62m.json` | 62,017,280 | 512 | OOM on this hardware; see below |

## The 61.7M target

`configs/boxz-62m.json` is exactly **62,017,280 parameters**
(d_model 640, 12 layers, 10 heads, d_ff 2560, vocab 4096, ctx 512).

It cannot be *trained* in this sandbox. Measured: a single forward+backward at
batch 1, ctx 512 peaks at **3,391 MB** of a **3,939 MB** / **0 swap** machine.
The trainer adds the corpus and the training loop and is OOM-killed (exit 137).

The config is kept because it is correct and portable — on a 16 GB machine it
trains unchanged. Reaching it here would need one of:

1. **Gradient checkpointing** — recompute activations in backward; ~30-40% less
   peak memory for ~33% more compute.
2. **CPU offload of AdamW state** — the optimiser's two moments are half the
   non-activation footprint.
3. **bf16/fp16 parameters** — halves weights and gradients.

None of the three is implemented; claiming otherwise would be untrue. The
honest position is: **the target is defined and configured, and is blocked by
3.9 GB of RAM, not by the code.**

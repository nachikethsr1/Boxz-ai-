# Boxz Development Roadmap

Boxz is a from-scratch AI language model. The roadmap is organised as **eight
major phases**, each delivering a substantial integrated capability rather than a
micro-step. This document states what is implemented, what is measured, and what
is deliberately deferred.

Legend: **✅ done** · **🟡 partial** · **⬜ not started**

---

## Phase 1 — Core AI Engine and Training Infrastructure ✅

| Deliverable | Status | Where |
|---|---|---|
| Project architecture & configuration system | ✅ | `boxz/`, `boxz/config.py`, `configs/` |
| Tensor & numerical utilities | ✅ | `boxz/engine/tensor.py`, `ops.py` |
| Embedding / Linear / LayerNorm / RMSNorm / Dropout | ✅ | `boxz/engine/layers.py` |
| Forward propagation | ✅ | `boxz/engine/tensor.py` |
| Backpropagation (reverse-mode autograd) | ✅ | `boxz/engine/tensor.py` |
| Loss calculation (cross entropy) | ✅ | `boxz/engine/ops.py` |
| Optimizer (AdamW + SGD) | ✅ | `boxz/engine/optim.py` |
| Training loop | ✅ | `boxz/training/trainer.py` |
| Dataset loading | ✅ | `boxz/data/` |
| Checkpoint save / load / resume | ✅ | `boxz/training/checkpoint.py` |
| Metrics & evaluation | ✅ | `boxz/training/evaluate.py` |
| Reproducible training | ✅ | seeded RNGs; `test_reproducible_training` |
| Finite-difference gradient checks | ✅ | `tests/test_tensor.py`, `tests/test_model.py` |

## Phase 2 — Tokenizer, Dataset Pipeline, Language Model ✅

| Deliverable | Status | Where |
|---|---|---|
| Byte-level BPE trainer (from scratch) | ✅ | `boxz/tokenizer/bpe.py` |
| Encoding / decoding / special tokens / lossless round-trip | ✅ | same |
| Vocabulary persistence, reproducible ids | ✅ | `BPETokenizer.save/load` |
| Text cleaning, dedupe, train/val split | ✅ | `boxz/data/text.py`, `corpus.py` |
| Sequence construction, context windows, batching | ✅ | `TokenizedCorpus` |
| Decoder-only Transformer (attention, MLP, residual, LN) | ✅ | `boxz/model/` |
| Autoregressive next-token prediction | ✅ | `BoxzLM.forward` |

## Phase 3 — Training, Checkpoints, Generation ✅

| Deliverable | Status | Where |
|---|---|---|
| Full LM training on real corpora | ✅ | `python -m boxz train` |
| LR scheduling, gradient clipping, init | ✅ | `boxz/engine/optim.py`, `config.py` |
| Checkpointing + exact resume | ✅ | `checkpoint.py`, `test_resume_continues_from_step` |
| Temperature / top-k / top-p / repetition control | ✅ | `boxz/inference/sampler.py` |
| Context-window management | ✅ | `generator.py`, `kv_cache.py` |
| Evaluation (loss, perplexity, bits-per-byte, accuracy) | ✅ | `evaluate.py`, `benchmark_suite.py` |

## Phase 4 — Conversational AI 🟡

| Deliverable | Status | Notes |
|---|---|---|
| Conversation formatting (`<|user|>`/`<|assistant|>`/`<|end|>`) | ✅ | special-token based |
| Message structure & history management | ✅ | `boxz/chat/format.py` |
| Multi-turn context with token budgeting | ✅ | `Conversation.truncated` |
| Instruction dataset + loader | ✅ | `data/sources/boxz_instruct.jsonl` (36 curated examples) |
| Instruction fine-tuning with assistant-only loss masking | ✅ | `boxz/training/sft.py` |
| Real conversational quality | 🟡 | **Limited by scale.** The SFT set is intentionally tiny; replies follow the chat format and the assistant voice, but factual breadth is inherited from a ~3 M-parameter base model trained on ~1 MB of text. |

Honest limitation: Boxz is a *real* language model, not a template bot, but at
this size its conversational competence is narrow. The infrastructure to scale
is what this phase delivers.

## Phase 5 — Coding Intelligence 🟡

| Deliverable | Status | Notes |
|---|---|---|
| Code-aware tokenizer (snake_case intact, lossless) | ✅ | |
| Instruction/code seed corpus (HTML, CSS, JS, Three.js, debugging) | ✅ | 36 examples |
| SFT pipeline ready for large code corpora | ✅ | `make_sft_batch`, masked loss |
| Trained coding capability at scale | ⬜ | Requires a large, licensed code corpus and much more compute than the dev machine provides. Not claimed. |

## Phase 6 — Advanced Game-Development Assistant ⬜

Infrastructure (multi-turn chat, KV-cache inference, SFT masking) is in place;
the specialised game-dev training data and evaluation suite are future work.
Seeded examples exist to establish the target format only.

## Phase 7 — Rust Performance Engine 🟡

| Deliverable | Status | Notes |
|---|---|---|
| Profiling before optimisation | ✅ | `boxz/bench/`, `docs/PERFORMANCE.md` |
| Python-side optimisations derived from profiling | ✅ | see Phase 1 perf notes |
| PyO3 crate (GELU, softmax, LayerNorm, sampling) | ✅ written | `rust/boxz_rust/` |
| Automated NumPy fallback | ✅ | `boxz/backends/` |
| Parity test harness | ✅ | `tests/test_backend_parity.py` |
| Compiled & benchmarked on this machine | ⬜ | No Rust toolchain / `static.rust-lang.org` unreachable in the build environment. Marked experimental. |

## Phase 8 — Complete Application 🟡

| Deliverable | Status | Where |
|---|---|---|
| Chat UI wired to the real engine | ✅ | `boxz/app/` |
| Model loading & metadata display | ✅ | `/api/status` |
| Generation settings | ✅ | temperature, top-k/top-p, max tokens, repetition |
| Conversation history & reset | ✅ | `/api/history`, `/api/reset` |
| Training status display | ✅ | `/api/metrics` + loss sparkline |
| Coding-oriented conversation mode | 🟡 | same engine; quality depends on Phase 5 |

---

## Immediate next steps (in priority order)

1. **Scale the base model** once more compute is available: 10–30 M parameters,
   `context_length` 512, a 10–100 MB licensed corpus. Config files already
   support this (`configs/`).
2. **Grow the instruction corpus** to thousands of examples and run a full SFT
   schedule; add held-out instruction evaluation thresholds.
3. **Compile the Rust backend** and re-run `tests/test_backend_parity.py` and
   `python -m boxz benchmark` on the target machine.
4. **Game-dev specialisation** (Phase 6): multi-file project datasets and
   task-level evaluation.

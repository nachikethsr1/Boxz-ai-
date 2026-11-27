# Boxz

**A language model built from scratch — no PyTorch, no pretrained weights, no external AI APIs.**

Boxz is a decoder-only Transformer language model whose *entire* learning system
is implemented in this repository: a reverse-mode autograd engine over NumPy, the
neural-network layers, causal self-attention, the AdamW optimiser, a byte-level
BPE tokenizer, the training loop, checkpointing, sampling and a chat application.
The only runtime dependency is NumPy.

The goal is a genuine AI assistant that eventually specialises in web and game
development (HTML/CSS/JavaScript/Three.js). This repository is the foundation:
a working model, a real training pipeline, and the instrumentation to prove both.

---

## What actually works today

| Capability | Status | Evidence |
|---|---|---|
| Custom autograd engine (≈30 differentiable ops) | ✅ verified by finite differences | `tests/test_tensor.py` |
| Byte-level BPE tokenizer, lossless round-trip | ✅ | `tests/test_tokenizer.py` |
| Decoder-only Transformer (causal MHA, GELU MLP, pre-norm) | ✅ | `tests/test_model.py` |
| Training loop with AdamW, cosine schedule, clipping, accumulation | ✅ | `tests/test_training.py` |
| Checkpoints + exact resume | ✅ | `test_resume_continues_from_step` |
| Text generation (temperature, top-k, top-p, repetition) | ✅ | `tests/test_inference.py` |
| KV-cache inference (verified identical logits) | ✅ | `test_kv_cache_matches_full_forward` |
| Instruction fine-tuning with assistant-only loss masking | ✅ | `boxz/training/sft.py` |
| Chat engine + web UI over the real model | ✅ | `tests/test_app.py` |
| Optional Rust kernels with NumPy fallback | 🟡 written, not compiled here | `rust/boxz_rust/` |

**Honest statement of scale.** The model checked in this milestone is a
**3.1 M-parameter** Transformer trained on ~1 MB of text (public-domain
Shakespeare plus a small curated instruction set). It is a real language model:
it learns syntax, word structure and style, and it continues prompts in
Shakespearean English. It is **not** yet a useful general assistant — that needs
a much larger corpus and more compute. The infrastructure to get there is what
this milestone delivers; see [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Quick start

```bash
# 1. environment (Python ≥ 3.9)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. fetch the public-domain corpus (declared in data/sources/*.json)
python -m boxz data tinyshakespeare

# 3. train a small model (~25–30 min on 4 CPU cores)
python -m boxz train --config configs/boxz-1m.json

# 4. sample from it
python -m boxz sample --checkpoint checkpoints/base-1m/best \
  --prompt "ROMEO:" --max-new-tokens 120 --temperature 0.8

# 5. chat in the browser
python -m boxz.app.server --checkpoint checkpoints/base-1m/best --port 8000
```

### Faster smoke test

```bash
python -m boxz train --config configs/smoke.json     # ~1 minute, proves the pipeline
```

---

## Project layout

```
boxz/
  engine/        autograd (tensor.py), differentiable ops, layers, AdamW, schedules
  tokenizer/     byte-level BPE implemented from scratch
  data/          registry, cleaning, splits, batching, instruction data
  model/         causal attention + Transformer language model
  training/      Trainer, checkpoints, evaluation, SFT, benchmark suite
  inference/     sampler, generator, KV cache
  chat/          conversation format and chat engine
  app/           HTTP server + chat UI (standard library only)
  backends/      NumPy reference backend + optional Rust loader
  bench/         micro-benchmarks and model-step benchmarks
rust/boxz_rust/  PyO3 extension (GELU, softmax, LayerNorm, sampling)
configs/         model/training configurations
data/sources/    dataset provenance: licence, URL, checksum, notes
docs/            architecture, training, performance, datasets, roadmap
tests/           133 tests: gradients, tokenizer, data, model, training,
                 inference, KV cache, chat, app, end-to-end
```

## Verify it works

```bash
python -m pytest tests/ -q                 # full suite
python -m pytest tests/ -q -m "not slow"   # fast subset (no training)
python -m boxz benchmark                   # micro-benchmarks
python -m boxz eval-suite --checkpoint checkpoints/base-1m/best
```

The suite includes:

* **finite-difference gradient checks** for every operation and for the whole
  model (one directional derivative over all parameters),
* a **causality test** proving a token cannot influence earlier positions,
* **checkpoint/resume** equality tests,
* a **training test** asserting validation loss actually falls,
* **KV-cache equivalence** tests against the plain forward pass,
* an **end-to-end** test that trains a model from raw text, saves it, reloads it
  and generates from it.

## Design principles

1. **No framework, no shortcuts.** The model, gradients, optimiser and sampler
   are ours. `numpy` is used exactly as a linear-algebra library.
2. **Measure before optimising.** Profiling found that GELU's `np.power`, `exp`
   subnormals and `np.add.at` were the real hotspots; each fix is documented with
   numbers in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
3. **No fake intelligence.** Every reply in the app comes from a trained
   checkpoint. There are no canned answers and no keyword routing.
4. **Everything reproducible.** A run's seed, architecture, data and optimiser
   state travel inside the checkpoint; a resumed run continues bit-identically.

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the engine and model work
* [docs/TRAINING.md](docs/TRAINING.md) — configs, hardware, reproduction
* [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — profiling results and optimisations
* [docs/DATASETS.md](docs/DATASETS.md) — licences and processing
* [docs/ROADMAP.md](docs/ROADMAP.md) — the eight phases and current status

## License

MIT for the code. Datasets keep their own licences (see `data/sources/*.json`).

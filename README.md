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
| Instruction fine-tuning (assistant-only masking, replay, held-out selection) | ✅ | `boxz/training/sft.py` |
| Token streaming over server-sent events | ✅ | `test_sse_endpoint_streams_and_terminates` |
| Chat engine + web UI over the real model | ✅ | `tests/test_app.py` |
| Licensing-audited code corpus fetcher | ✅ | `boxz/data/code_corpus.py` |
| Optional Rust kernels with NumPy fallback | 🟡 written, not compiled here | `rust/boxz_rust/` |

## Measured results

Not projections — every figure below comes from the checkpoints in
`checkpoints/` and is reproducible with the commands in
[docs/EVALUATION.md](docs/EVALUATION.md).

| | Pretrained base | After instruction tuning |
|---|---|---|
| parameters | 3,087,360 | 3,087,360 |
| training corpus | 934 K tokens (Shakespeare + MIT three.js/jQuery code) | + 100 instruction examples |
| validation loss | **4.0353** (ppl 56.6) | 4.1445 (ppl 63.1) |
| next-token accuracy | 0.244 | 0.210 |
| **held-out instruction loss** | 7.5165 | **4.9116** (−34.6 %) |
| instruction-format rate | 0.00 | **0.33** |
| generation speed | ~89 tok/s | ~92 tok/s |

**What this means, stated plainly.** Boxz is a genuine 3-million-parameter
Transformer with its own autograd engine, tokenizer, optimiser and training
loop — no PyTorch, no pretrained weights, no API. Trained on under a million
tokens of text and code, it produces syntactically shaped English, HTML, CSS
and Three.js, and after instruction tuning it answers in the assistant voice and
stops at the end of a turn. It does **not** reliably answer programming
questions correctly: with 3 M parameters over 934 K tokens it composes phrases
from the domains it saw and then tends to repeat. That is the honest limit of
this scale, and it is why the next phase is more data and more parameters rather
than prompt engineering. See [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Quick start

```bash
# 1. environment (Python ≥ 3.9)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. fetch the public-domain corpus (declared in data/sources/*.json)
python -m boxz data tinyshakespeare

# 3. fetch the licensed code corpus (three.js + jQuery, MIT-verified)
python -m boxz.data.code_corpus fetch --max-files 400

# 4. train (~60 min on 4 CPU cores; --max-minutes caps it)
python -m boxz train --config configs/boxz-1m.json

# 5. instruction-tune (~13 min, selects its own best checkpoint)
python -m boxz sft --checkpoint checkpoints/base-1m/best --out checkpoints/sft

# 6. evaluate both honestly
python -m boxz eval-suite --checkpoint checkpoints/sft/best --out reports/sft.json

# 7. sample from it
python -m boxz sample --checkpoint checkpoints/base-1m/best \
  --prompt "ROMEO:" --max-new-tokens 120 --temperature 0.8

# 8. chat in the browser (streaming)
python -m boxz.app.server --checkpoint checkpoints/sft/best --port 8000
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
  data/          registry, cleaning, splits, batching, instruction data,
                 licensed code-corpus fetcher, instruction-set builder
  model/         causal attention + Transformer language model
  training/      Trainer, checkpoints, evaluation, SFT (replay + holdout), suite
  inference/     sampler, generator, KV cache
  chat/          conversation format and chat engine
  app/           HTTP server + chat UI (standard library only)
  backends/      NumPy reference backend + optional Rust loader
  bench/         micro-benchmarks and model-step benchmarks
rust/boxz_rust/  PyO3 extension (GELU, softmax, LayerNorm, sampling)
configs/         model/training configurations
data/sources/    dataset provenance: licence, URL, checksum, notes
docs/            architecture, training, performance, datasets, roadmap
tests/           138 tests: gradients, tokenizer, data, model, training,
                 inference, KV cache, chat, streaming, app, code corpus,
                 end-to-end
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

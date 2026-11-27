# Training Boxz

Everything below runs on a CPU. The reference machine is an i5-1135G7 (4 cores,
8 GB RAM) with no discrete GPU; the numbers here were measured there.

## 1. Prepare

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m boxz data list               # datasets + licences
python -m boxz data tinyshakespeare    # public-domain corpus -> data/raw/
```

The data stage is deterministic: cleaning, de-duplication and the train/val split
are reproducible, and the trained tokenizer is cached under `data/processed/` so
restarts are instant.

## 2. Pretrain

```bash
python -m boxz train --config configs/boxz-1m.json
```

`configs/boxz-1m.json` — the model used in this milestone:

| | |
|---|---|
| parameters | 3.09 M |
| layers / heads / width | 6 / 6 / 192 |
| context length | 128 |
| vocabulary | 2048 (byte-level BPE) |
| corpus | Tiny Shakespeare + Boxz instructions (≈1.05 MB) |
| batch × context | 24 × 128 (3 072 tokens/step) |
| steps | 1200 (≈3.7 M tokens) |
| optimiser | AdamW, lr 1.2e-3 → 1.2e-4 cosine, warmup 100, wd 0.01, clip 1.0 |
| wall clock | ≈35–40 min on 4 CPU cores (≈1.8–2.2 s/step) |

Anything can be overridden on the command line:

```bash
python -m boxz train --config configs/boxz-1m.json \
  --set model.n_layer=8 --set model.d_model=256 --set max_steps=3000
```

### Checkpoints

```
checkpoints/base-1m/
  train_log.jsonl     JSONL, one record per logged step (loss, lr, tokens, eval)
  best/               lowest validation loss so far
  step_300/ ...       periodic
  final/              last step
```

Each checkpoint directory holds `model.npz`, `config.json`, `tokenizer.json`,
`train_state.json` and `optimizer.npz`. Resuming is exact:

```bash
python -m boxz train --config configs/boxz-1m.json \
  --set resume=checkpoints/base-1m/step_300 --set max_steps=1600
```

## 3. Instruction fine-tuning

```bash
python -m boxz sft --checkpoint checkpoints/base-1m/best \
  --dataset data/sources/boxz_instruct.jsonl --steps 400
```

This trains only on assistant tokens: prompt labels are replaced by `<pad>` and
ignored by the loss, so the model learns the conversation format and the
assistant voice rather than echoing the question. Default hyper-parameters:
lr 2e-4 cosine, batch 8, max length 256, 400 steps.

## 4. Evaluate

```bash
python -m boxz eval --checkpoint checkpoints/base-1m/best        # loss / ppl / bpb
python -m boxz eval-suite --checkpoint checkpoints/base-1m/best  # capability report
```

`eval-suite` reports validation loss, perplexity, bits-per-byte, next-token
accuracy, instruction-format compliance, instruction loss on a held-out slice
and generation throughput, plus sample continuations.

## 5. Choose a size for your hardware

Measured throughput (tokens/second, higher is better) for candidate shapes on the
reference 4-core CPU:

| d_model | layers | ctx | batch | params | tok/s |
|---|---|---|---|---|---|
| 192 | 4 | 128 | 32 | 2.59 M | ≈3 100 |
| 192 | 6 | 128 | 24 | 3.09 M | ≈1 900 |
| 256 | 6 | 128 | 24 | 5.30 M | ≈1 650 |

Budget roughly `tokens_needed ≈ 20 × parameters` for a compute-optimal run; at
2 000 tok/s that is ≈8.5 hours for the 3 M-parameter model. For a comfortable
overnight run, prefer ~10–30 M tokens with a 5–10 M-parameter model.

To scale to a bigger model, add a config (see `configs/`) and keep the same
commands. Memory is the binding constraint on 8 GB: activations for a step are
`≈ batch × context × d_model × 4 bytes × (10–20)`.

## 6. Reproducibility

* Every run is seeded (`TrainConfig.seed`); weight init, batch order and dropout
  all derive from it, and `test_reproducible_training` asserts identical loss
  traces across runs.
* Checkpoints embed the full model and training configuration.
* Logs are append-only JSONL, so a run can be audited after the fact.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ValueError: context_length` | sequence longer than the model was built for; raise `model.context_length` and rebuild, or shorten the prompt |
| Loss stays at `ln(vocab)` | learning rate too low or data not reaching the model; check `tokens_seen` in the log |
| Loss becomes `nan` | lower `lr`, keep `grad_clip` ≤ 1.0, verify the data has no binary junk |
| Very slow first step | the BPE tokenizer is being trained; subsequent runs use the cache in `data/processed/` |
| `KeyError: unknown dataset` | add a JSON file under `data/sources/` or pass a path in `data.dataset` |

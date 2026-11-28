# Checkpoints

Trained checkpoints are **not** committed to git (they are 12-100 MB binaries and
would grow the repository without benefit).

Reproduce them locally:

```bash
pip install -r requirements.txt
python -m boxz train --config configs/boxz-1m.json      # base model  (~30 min CPU)
python -m boxz sft --checkpoint checkpoints/base-1m/best # instruction tuning (~5 min)
```

Directory layout produced by training:

```
checkpoints/base-1m/
  train_log.jsonl         one JSON object per logged step
  best/                   best validation checkpoint (model.npz, config.json, tokenizer.json, ...)
  step_300/ ...           periodic checkpoints
  final/                  last checkpoint
checkpoints/sft/
  final/                  instruction-tuned model (+ sft_report.json)
```

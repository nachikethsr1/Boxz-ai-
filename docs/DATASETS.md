# Datasets

Boxz never downloads data silently. Every dataset is declared in
`data/sources/<name>.json` with its licence, source URL, format and SHA-256, and
is fetched only by an explicit command (`python -m boxz data <name>`).

## Declared datasets

| Name | Corpus | Licence | Size | Role |
|---|---|---|---|---|
| `tinyshakespeare` | ~1.1 MB of Shakespeare's plays, extracted from `char-rnn` (`data/tinyshakespeare/input.txt`) | **Public domain** text; the hosting repository is MIT | 1 115 394 bytes | Pretraining corpus — small, clean, fast on CPU |
| `boxz_instruct` | 36 curated instruction/response pairs (identity, HTML/CSS, JavaScript, debugging, Three.js, game design, ML fundamentals) | **CC0-1.0**, written for this project | ~28 KB | Tokenizer coverage for code/chat markers + seed set for SFT |

Both are declared with SHA-256 checksums where a download is involved.

## Why so little data?

The reference machine has 4 CPU cores and 8 GB RAM. A 3 M-parameter model needs
~20× its parameter count in tokens (~60 M) for a compute-optimal run; the
corpus above provides ~0.35 M training tokens, which is enough to demonstrate
(wavenue) learning but not enough for broad competence. Scaling the corpus is a
data-provisioning decision, not an architectural one — the pipeline, tokenizer
and trainer are already size-agnostic.

## Adding a dataset

Create `data/sources/<name>.json`:

```json
{
  "title": "My corpus",
  "licence": "CC-BY-4.0",
  "url": "https://codeload.github.com/owner/repo/tar.gz/refs/heads/main",
  "filename": "mycorpus.txt",
  "fmt": "text",
  "sha256": "<optional>",
  "extract_member": "repo-main/data/mycorpus.txt",
  "notes": "What it contains, how it was produced, what it is used for.",
  "allowed_use": "Attribution required; commercial use permitted."
}
```

Then:

```bash
python -m boxz data mycorpus       # downloads + verifies
```

and reference it from a config:

```json
"data": { "dataset": ["tinyshakespeare", "mycorpus"], "vocab_size": 4096 }
```

For instruction data the supported JSONL shapes are:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "response": "..."}
```

Multi-turn conversations are supported; the last message must be from the
assistant.

## Processing

* Text: Unicode NFC normalisation, control-character removal, whitespace
  normalisation, optional line-level de-duplication (first occurrence kept),
  optional truncation — all deterministic (`boxz/data/text.py`).
* Chat examples embedded in a pretraining corpus are rendered with the model's
  chat markers so the base model has seen the format before SFT.
* Splits: a contiguous tail split (`val_fraction`, default 5 %) so validation
  windows never overlap training windows.
* Encoding: one byte-level BPE vocabulary per run, trained on up to 4 M
  characters, cached in `data/processed/` with the corpus statistics.

## Licensing policy

* Only public-domain, permissively licensed, or project-authored data.
* No private user data, credentials or scraped personal information.
* Large corpora and checkpoints are **not** committed (`checkpoints/`,
  `data/raw/`, `data/processed/` are git-ignored); provenance is committed
  instead.

# Datasets

Boxz never downloads data silently. Every dataset is declared in
`data/sources/<name>.json` with its licence, source URL, format and SHA-256, and
is fetched only by an explicit command (`python -m boxz data <name>`).

## Declared datasets

| Name | Corpus | Licence | Size | Role |
|---|---|---|---|---|
| `tinyshakespeare` | ~1.1 MB of Shakespeare's plays, extracted from `char-rnn` (`data/tinyshakespeare/input.txt`) | **Public domain** text; the hosting repository is MIT | 1 115 394 bytes | English prose for pretraining |
| `boxz_web_code` | 473 web-game source files: `three.js` `examples/*.html` (400) and `jquery` `src/*.js` (73), cleaned of vendored CDN includes | **MIT** (verified through the GitHub API at fetch time) | 2 717 338 chars | HTML/CSS/JS/Three.js for pretraining — the model cannot write these languages without seeing them |
| `boxz_instruct` | 36 curated instruction/response pairs, hand-written seed set | **CC0-1.0**, project original | 14.7 KB | Seed set for SFT / format reference |
| `boxz_instruct_expanded` | 116 examples: the seed set plus paraphrase groups with genuinely different answers (identity, tokens/attention, HTML, CSS, JavaScript, canvas, animation, Three.js, collisions, audio, debugging, game design, ML basics) | **CC0-1.0**, project original | 51 KB | SFT training set |

### Code corpus provenance

`python -m boxz.data.code_corpus fetch` verifies each project's licence through
`api.github.com` **before downloading a single byte**, and refuses the project on
a mismatch. It then records, in `data/raw/code_corpus/manifest.json`:

* the SPDX licence GitHub reports for the repository,
* the exact commit SHA that was harvested,
* every file path, its byte count and its licence.

The exact commit SHA is pinned for every file, so a run can be reproduced or
withdrawn later. The fetched corpus itself is **not** committed — only the
fetcher and the declared source are.

### Held-out separation (important)

The SFT evaluation rows are withheld from *every* stage:

* `boxz/data/corpus.py` drops the rows selected by
  `boxz.training.sft.holdout_indices` when the instruction file is read as a
  pretraining source (`DataConfig.exclude_sft_holdout`, on by default).
* The SFT trainer never fits those rows and selects its checkpoint on their
  loss (`best/`), with early stopping.
* The benchmark suite reports instruction loss on the same rows.

`tests/test_data.py::test_pretraining_corpus_excludes_sft_holdout` asserts both
directions, including a positive control that the guard actually removes rows.

## Why so little data?

The reference machine has 4 CPU cores and 8 GB RAM. A 3 M-parameter model needs
roughly 20× its parameter count in tokens (~60 M) for a compute-optimal run.
The current mix provides **~0.89 M training tokens** (934 K including
validation): enough for coherent Shakespeare-style English and basic
HTML/CSS/JavaScript patterns, nowhere near enough for broad competence. Scaling
is a data-provisioning decision, not an architectural one — the tokenizer,
trainer and pipeline are size-agnostic, and the `--max-files` flag on the code
fetcher is the single knob for a bigger corpus.

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

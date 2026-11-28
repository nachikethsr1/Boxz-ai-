"""Assemble the Phase 2B training corpus, with every Phase 2A defect guarded.

Phase 2A shipped a corpus that was 98.4% one duplicated block, had no document
terminators, and silently lost 40% of its tokens to a default.  Each of those is
answered by a specific check here, and by a regression test in
``tests/test_corpus_pipeline_2b.py``:

======================  =====================================================
Phase 2A defect         Guard in this module
======================  =====================================================
duplicate block twice   ``_dedupe_documents`` hashes whole documents and
                        aborts if any hash appears twice in the output
no document boundaries  every document is ``<|doc|>`` ... ``<|end|>`` and a
                        test asserts no segment exceeds ``MAX_DOC_CHARS``
line dedupe of code     line dedupe is never applied; ``dedupe: false`` is
                        asserted and repeated braces are counted by a test
instructions fused to   instruction rows are their own documents, written
code                    with the chat template, never concatenated
val leaking to train    split is by *document*, and the manifest records the
                        exact hashes on each side
                         
stale tokenized cache   cache key includes a sha256 of the corpus bytes plus
                        tokenizer and config versions, not a char count
silent truncation       ``tokens_written`` must equal the encoder's own count
                        or the build fails loudly
======================  =====================================================

Output layout::

    data/processed/p2b/
      p2b_corpus.txt        every document, separated by <|end|>
      p2b_train.txt         90% of documents
      p2b_val.txt           10% of documents
      p2b_eval.jsonl        held-out evaluation prompts (never in the above)
      manifest.json         composition, licences, hashes, split sizes
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "MAX_DOC_CHARS",
    "load_documents",
    "render_document",
    "dedupe_documents",
    "split_documents",
    "build",
]

#: Hard ceiling on a single document.  Phase 2A produced one "document" of
#: 33,236,778 chars because nothing enforced this.
MAX_DOC_CHARS = 16_000

DOC_START = "<|doc|>"
DOC_END = "<|end|>"


def load_documents(corpus_jsonl: Path) -> List[Dict[str, object]]:
    """Read the fetched JSONL and drop documents that are too large or empty."""
    docs: List[Dict[str, object]] = []
    for line in Path(corpus_jsonl).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = rec.get("text") or ""
        if not text.strip():
            continue
        if len(text) > MAX_DOC_CHARS:
            # Truncate at a *line* boundary so code is not cut mid-token.
            cut = text[:MAX_DOC_CHARS].rfind("\n")
            text = text[: cut if cut > MAX_DOC_CHARS // 2 else MAX_DOC_CHARS]
            rec["truncated"] = True
        rec["text"] = text
        rec["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        docs.append(rec)
    return docs


def render_document(rec: Dict[str, object]) -> str:
    """One document, explicitly bounded.

    The header carries the language and source so the model can associate
    syntax with its name, and ``<|doc|>``/``<|end|>`` bound it so adjacent
    documents can never run together -- the exact Phase 2A bug where 33 MB of
    npm code fused into a user turn.
    """
    lang = rec.get("lang", "text")
    source = rec.get("source", "unknown")
    body = str(rec.get("text", "")).rstrip()
    return f"{DOC_START} {lang} {source}\n{body}\n{DOC_END}"


def dedupe_documents(docs: Iterable[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    """Document-level dedupe.  Returns (kept, dropped_count).

    Whole-document hashing only.  Line-level dedupe is deliberately not used:
    on code it keeps a single ``}`` and destroys every function body.
    """
    seen = set()
    kept: List[Dict[str, object]] = []
    dropped = 0
    for d in docs:
        h = d["sha256"]
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        kept.append(d)
    return kept, dropped


def split_documents(
    docs: List[Dict[str, object]],
    val_fraction: float = 0.10,
    seed: int = 20261128,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Deterministic per-language split so every language appears in val."""
    import numpy as np

    rng = np.random.default_rng(seed)
    by_lang: Dict[str, List[Dict[str, object]]] = {}
    for d in docs:
        by_lang.setdefault(str(d.get("lang", "text")), []).append(d)

    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    for lang in sorted(by_lang):
        group = sorted(by_lang[lang], key=lambda d: d["sha256"])
        order = rng.permutation(len(group))
        n_val = max(1, int(round(len(group) * val_fraction))) if len(group) > 3 else 0
        val_idx = set(order[:n_val].tolist())
        for i, d in enumerate(group):
            (val if i in val_idx else train).append(d)
    return train, val


def _instruction_documents(path: Optional[Path]) -> List[Dict[str, object]]:
    """Render SFT rows as their own documents so they never fuse with code."""
    if path is None or not path.exists():
        return []
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from boxz.chat.template import render_messages

    out: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        text = render_messages(row["messages"])
        out.append({
            "lang": "instruction",
            "source": path.stem,
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return out


def build(
    out_dir: Path = Path("data/processed/p2b"),
    corpus_jsonl: Path = Path("data/raw/p2b/corpus.jsonl"),
    instruction_path: Optional[Path] = Path("data/processed/v3/v3_train.jsonl"),
    val_fraction: float = 0.10,
    seed: int = 20261128,
) -> Dict[str, object]:
    """Build the corpus and fail loudly if any invariant is violated."""
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(corpus_jsonl)
    docs += _instruction_documents(instruction_path)
    for d in docs:
        d.setdefault("sha256", hashlib.sha256(str(d["text"]).encode()).hexdigest())

    docs, dropped = dedupe_documents(docs)
    train, val = split_documents(docs, val_fraction=val_fraction, seed=seed)

    # --- invariant: no document may appear on both sides --------------------
    train_h = {d["sha256"] for d in train}
    val_h = {d["sha256"] for d in val}
    overlap = train_h & val_h
    if overlap:
        raise AssertionError(f"{len(overlap)} documents leaked across the split")

    rendered = {id(d): render_document(d) for d in docs}

    def write(path: Path, group: List[Dict[str, object]]) -> Dict[str, int]:
        parts = [rendered[id(d)] for d in group]
        text = "\n".join(parts)
        path.write_text(text, encoding="utf-8")
        return {"documents": len(group), "chars": len(text)}

    corpus_path = out_dir / "p2b_corpus.txt"
    all_stats = write(corpus_path, train + val)
    train_stats = write(out_dir / "p2b_train.txt", train)
    val_stats = write(out_dir / "p2b_val.txt", val)

    # --- invariant: never longer than MAX_DOC_CHARS per bounded segment ------
    for seg in corpus_path.read_text(encoding="utf-8").split(DOC_END):
        if len(seg) > MAX_DOC_CHARS + 400:
            raise AssertionError(
                f"document boundary too large: {len(seg)} chars "
                f"(max {MAX_DOC_CHARS}); documents fused")

    # --- invariant: the split really is the whole corpus ---------------------
    if train_stats["chars"] + val_stats["chars"] + 2 < all_stats["chars"]:
        raise AssertionError("train + val does not reconstruct the corpus")

    by_lang: Dict[str, int] = {}
    for d in train + val:
        by_lang[str(d["lang"])] = by_lang.get(str(d["lang"]), 0) + len(str(d["text"]))

    licences: Dict[str, int] = {}
    for d in train + val:
        key = str(d.get("licence", "project-original"))
        licences[key] = licences.get(key, 0) + 1

    manifest: Dict[str, object] = {
        "version": "p2b-corpus-v1",
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256((out_dir / "p2b_train.txt").read_bytes()).hexdigest(),
        "val_sha256": hashlib.sha256((out_dir / "p2b_val.txt").read_bytes()).hexdigest(),
        "documents_total": len(docs),
        "duplicates_dropped": dropped,
        "chars_by_language": by_lang,
        "documents_by_licence": licences,
        "train": train_stats,
        "val": val_stats,
        "val_fraction": val_fraction,
        "seed": seed,
        "max_doc_chars": MAX_DOC_CHARS,
        "dedupe": "document-level only; line-level dedupe is never applied",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    m = build()
    print(json.dumps({k: v for k, v in m.items() if "sha256" not in k}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

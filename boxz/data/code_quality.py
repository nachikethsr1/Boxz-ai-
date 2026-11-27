"""Corpus quality filtering for the web-code corpus.

Diagnosis this addresses
------------------------
The raw three.js slice is 7.3 % HTML metadata and 11.6 % boilerplate lines that
appear in 50+ files, and the later examples are near-duplicate copies.  A 3 M
parameter model trained on that spends a large share of its capacity on
``<meta property="og:title" ...>`` and on copying the same
``renderer.setSize( window.innerWidth, window.innerHeight );`` line.  When it
then tries to "write a minimal HTML page" it reproduces the metadata header
instead of a document.

Filtering steps (all deterministic, no model involved):

1. Drop noise tags: ``<meta>``, ``<link>``, ``<title>`` and the importmap.
2. Drop lines that are boilerplate: appearing in >= ``min_repeats`` files.
3. Drop files whose remaining content is below a threshold.
4. Near-duplicate removal on 5-token shingles, keeping the first (avoids
   training on 400 near-identical examples).
5. Cap each file so one 12 KB page cannot dominate a batch.

    python -m boxz.data.code_quality --report
    python -m boxz.data.code_quality --clean --out data/raw/code_corpus/clean.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = ["FilterConfig", "clean_documents", "quality_report", "main"]

#: Lines we never want the model to learn to open a response with.
_NOISE_TAGS = re.compile(
    r"<meta\b[^>]*>|<link\b[^>]*>|<title[^>]*>.*?</title>"
    r"|<script[^>]+type=[\"']importmap[\"'][^>]*>.*?</script>",
    re.IGNORECASE | re.DOTALL,
)
_BLANK_RUNS = re.compile(r"\n[ \t]*\n[ \t]*\n+")


@dataclass
class FilterConfig:
    min_repeats: int = 40        # drop a line present in >= N files
    min_chars: int = 600         # drop files smaller than this after cleaning
    max_chars: int = 6000        # cap per file so a 12 KB page cannot dominate
    shingle_size: int = 5        # near-duplicate detection window (tokens)
    dup_threshold: float = 0.85  # jaccard on shingles above which = duplicate
    drop_noise_tags: bool = True


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+|[^\s\w]", text.lower())


def _shingles(tokens: Sequence[str], k: int) -> set:
    if len(tokens) < k:
        return set(tokens)
    return {hashlib.blake2b(" ".join(tokens[i:i + k]).encode(), digest_size=8).digest()
            for i in range(len(tokens) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _strip_noise(code: str) -> str:
    code = _NOISE_TAGS.sub("", code)
    code = _BLANK_RUNS.sub("\n\n", code)
    return "\n".join(ln.rstrip() for ln in code.splitlines()).strip()


def _line_counts(docs: Sequence[dict]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for d in docs:
        for ln in set(d["code"].splitlines()):
            s = ln.strip()
            if len(s) > 20:
                counts[s] += 1
    return counts


def clean_documents(docs: Sequence[dict], cfg: Optional[FilterConfig] = None,
                    verbose: bool = True) -> List[dict]:
    """Return a filtered, de-duplicated copy of the corpus document list."""
    cfg = cfg or FilterConfig()
    counts = _line_counts(docs)
    boiler = {ln for ln, n in counts.items() if n >= cfg.min_repeats}

    kept: List[dict] = []
    kept_shingles: List[set] = []
    stats = collections.Counter()

    for d in docs:
        code = _strip_noise(d["code"]) if cfg.drop_noise_tags else d["code"]
        stats["noise_chars_removed"] += len(d["code"]) - len(code)

        lines = [ln for ln in code.splitlines() if ln.strip() not in boiler]
        code = "\n".join(lines).strip()
        if len(code) < cfg.min_chars:
            stats["dropped_short"] += 1
            continue

        sh = _shingles(_tokens(code[:4000]), cfg.shingle_size)
        if any(_jaccard(sh, prev) >= cfg.dup_threshold for prev in kept_shingles):
            stats["dropped_duplicate"] += 1
            continue

        kept.append({**d, "code": code[:cfg.max_chars], "chars": min(len(code), cfg.max_chars)})
        kept_shingles.append(sh)

    if verbose:
        print(f"[quality] input {len(docs)} files / {sum(d['chars'] for d in docs):,} chars")
        print(f"[quality] removed {stats['noise_chars_removed']:,} chars of meta/link/title")
        print(f"[quality] dropped {stats['dropped_short']} short files, "
              f"{stats['dropped_duplicate']} near-duplicates, "
              f"{len(boiler)} boilerplate lines")
        print(f"[quality] kept  {len(kept)} files / {sum(d['chars'] for d in kept):,} chars")
    return kept


def quality_report(path: str = "data/raw/code_corpus/corpus.json",
                   cfg: Optional[FilterConfig] = None) -> dict:
    cfg = cfg or FilterConfig()
    docs = json.loads(Path(path).read_text(encoding="utf-8"))
    cleaned = clean_documents(docs, cfg, verbose=False)
    counts = _line_counts(docs)
    before = sum(d["chars"] for d in docs)
    after = sum(d["chars"] for d in cleaned)
    return {
        "files_before": len(docs), "files_after": len(cleaned),
        "chars_before": before, "chars_after": after,
        "reduction": round(1 - after / max(before, 1), 3),
        "boilerplate_lines": sum(1 for _, n in counts.items() if n >= cfg.min_repeats),
        "top_boilerplate": [ln for ln, n in counts.most_common(5) if n >= cfg.min_repeats],
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Filter and de-duplicate the code corpus")
    p.add_argument("--corpus", default="data/raw/code_corpus/corpus.json")
    p.add_argument("--out", default="data/raw/code_corpus/clean.json")
    p.add_argument("--report", action="store_true")
    p.add_argument("--clean", action="store_true")
    args = p.parse_args(argv)

    if args.report or not args.clean:
        print(json.dumps(quality_report(args.corpus), indent=2))
    if args.clean:
        docs = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
        cleaned = clean_documents(docs)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
        print(f"[quality] wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

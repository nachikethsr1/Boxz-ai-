"""Line-level deduplication for the code corpus.

The cleaned corpus still trains the model on the same line hundreds of times:
``renderer.setSize( window.innerWidth, window.innerHeight );`` appeared in 659 of
473 files.  Frequency that high teaches repetition, and the generated output
showed exactly that (``renderer, renderer, renderer``).

This module keeps the *first* occurrence of each exact line across the whole
corpus and drops later copies, so a line is learned once no matter how many
tutorial pages happen to contain it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["dedupe_documents", "main"]

SRC = "data/raw/code_corpus/clean.json"
DST = "data/raw/code_corpus/dedup.json"


def dedupe_documents(docs: List[dict], min_line: int = 8) -> List[dict]:
    """Keep first occurrence of each exact line corpus-wide."""
    seen: set = set()
    out: List[dict] = []
    removed = 0
    for d in docs:
        kept: List[str] = []
        for line in d["code"].splitlines():
            s = line.strip()
            if len(s) >= min_line:
                if s in seen:
                    removed += 1
                    continue
                seen.add(s)
            kept.append(line)
        code = "\n".join(kept).strip()
        if len(code) >= 400:
            out.append({**d, "code": code, "chars": len(code)})
    total_in = sum(d["chars"] for d in docs)
    total_out = sum(d["chars"] for d in out)
    print(f"[dedupe] {len(docs)} -> {len(out)} docs, "
          f"{total_in:,} -> {total_out:,} chars ({1-total_out/total_in:.1%} smaller), "
          f"{removed:,} repeated lines removed")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="boxz.data.dedupe_corpus")
    ap.add_argument("--in", dest="src", default=SRC)
    ap.add_argument("--out", dest="dst", default=DST)
    args = ap.parse_args(argv)
    docs = json.loads(Path(args.src).read_text(encoding="utf-8"))
    out = dedupe_documents(docs)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[dedupe] wrote {dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

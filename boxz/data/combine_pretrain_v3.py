"""Combine the npm code corpus with the rendered instruction rows.

Order matters for the loss curve: instruction rows go **last** so that the
model's final pretraining glimpses are the exact chat-template format SFT will
use.  The npm code (13.7M tokens) provides the syntax prior; the 265k
instruction tokens provide the format prior.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
from pathlib import Path

__all__ = ["combine"]


def combine(npm_glob: str, sft_text: Path, out: Path) -> dict:
    npm_files = sorted(glob.glob(npm_glob))
    if not npm_files:
        raise SystemExit(f"no files matched {npm_glob}")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with io.open(out, "w", encoding="utf-8") as fh:
        for f in npm_files:
            fh.write(Path(f).read_text(encoding="utf-8"))
            fh.write("\n\n")
            n += 1
        if sft_text.exists():
            fh.write(sft_text.read_text(encoding="utf-8"))
    return {"npm_files": n, "output": str(out), "bytes": out.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npm-glob", default="data/raw/npm/*.txt")
    ap.add_argument("--sft", type=Path, default=Path("data/processed/v3/pretrain_sft.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/v3/pretrain.txt"))
    a = ap.parse_args(argv)
    print(json.dumps(combine(a.npm_glob, a.sft, a.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

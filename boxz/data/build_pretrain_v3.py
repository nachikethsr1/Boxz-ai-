"""Build the V3 pretraining corpus from the SFT rows plus any local web-code text.

Pretraining teaches the model the *distribution* of web text and code; SFT then
teaches it to answer.  With only 4,104 instruction rows we would otherwise
pretrain on the same text twice, which overfits.

The corpus is therefore::

    every SFT turn rendered through the real chat template   (in-domain)
  + every raw code/markdown fence body in the rows           (code density)
  + data/raw/code_corpus/*.txt and data/raw/*.txt if present (extra prose)

Rows whose question is in the SFT **validation** split are excluded, so the
val loss stays an honest measurement.

CLI:  python -m boxz.data.build_pretrain_v3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..chat.template import render_messages

__all__ = ["build_pretrain", "read_sft_rows"]

_FENCE_BODY = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def read_sft_rows(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_pretrain(train_jsonl: Path, out_path: Path,
                   extra_dirs: tuple = ("data/raw",)) -> dict:
    rows = read_sft_rows(train_jsonl)
    chunks: list[str] = []

    for r in rows:
        chunks.append(render_messages(r["messages"]))
        # code bodies on their own increase code density per token
        for m in r["messages"]:
            if m["role"] == "assistant":
                for body in _FENCE_BODY.findall(m["content"]):
                    chunks.append(m["content"].split("```")[0].strip() or "")
                    chunks.append(body)

    extras = []
    for d in extra_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.txt")) + sorted(p.rglob("*.md")):
            if f.name in ("README.md",) and f.parent == Path("."):
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(txt) > 200:
                extras.append(txt)
                chunks.append(txt)

    text = "\n\n".join(c for c in chunks if c and c.strip())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return {
        "sft_rows": len(rows),
        "extra_files": len(extras),
        "chars": len(text),
        "output": str(out_path),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/processed/v3/v3_train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/v3/pretrain.txt"))
    args = ap.parse_args(argv)
    print(json.dumps(build_pretrain(args.train, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Side-by-side responses from two checkpoints on a fixed prompt set.

Used to answer the question the task actually asks: *did the retrain change what
the model says?*  Prompts are fixed so the comparison is reproducible, and the
same generation settings are used for both sides.

CLI::

    python -m boxz.eval.compare --before checkpoints/old/best \\
                                --after  checkpoints/boxz-26m/boxz-26m/best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

__all__ = ["PROMPTS", "compare", "main"]

#: Deliberately mixes maths (engine-answered), coding, and open prose.
PROMPTS: List[str] = [
    # maths -- must be exact regardless of the model
    "482736 + 918264",
    "What is 15% of 240?",
    "solve 4x + 7 = 31",
    # coding
    "Write JavaScript that responds to a button click.",
    "Write HTML for a simple game page.",
    "Write CSS to centre a box.",
    "Debug this: document.getElementById('btn').addEventListener('click', go);"
    " function go( { console.log('x'); }",
    # concept
    "What is HTML?",
    "Explain what CSS does.",
    # topic switch
    "Now give me a math answer instead: 7 * 8",
]


def _load(checkpoint: str):
    from ..chat.engine import ChatEngine
    from ..config import GenerationConfig
    from ..training.checkpoint import load_checkpoint

    ck = load_checkpoint(checkpoint)
    eng = ChatEngine(ck.model, ck.tokenizer, use_tools=True)
    cfg = GenerationConfig(max_new_tokens=140, temperature=0.4, top_k=20, top_p=0.95)
    return eng, cfg


def compare(before: Optional[str], after: str, out: Path) -> dict:
    eng_a, cfg = _load(after)
    eng_b = None
    if before and Path(before).exists():
        eng_b, _ = _load(before)

    rows = []
    for p in PROMPTS:
        after_reply = eng_a.reply(p, cfg)
        eng_a.reset()
        row = {"prompt": p, "after": after_reply.text,
               "after_finish": after_reply.finish_reason,
               "after_tools": after_reply.tools}
        if eng_b is not None:
            before_reply = eng_b.reply(p, cfg)
            eng_b.reset()
            row["before"] = before_reply.text
            row["before_finish"] = before_reply.finish_reason
        rows.append(row)

    result = {"before": before, "after": after, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    for r in rows:
        print("=" * 70)
        print("PROMPT:", r["prompt"])
        if "before" in r:
            print("  BEFORE:", r["before"][:220].replace("\n", " "))
        print("  AFTER :", r["after"][:220].replace("\n", " "))
        print("  finish:", r.get("before_finish", "-"), "->", r["after_finish"])
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", default="reports/compare.json")
    a = ap.parse_args(argv)
    compare(a.before, a.after, Path(a.out))
    print(f"\nwritten -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

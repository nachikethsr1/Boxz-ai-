"""Clean a user-supplied ``Data.txt`` and turn it into SFT training data.

Measured problems in the first uploaded file (2,000 conversations):

* **152 tokens of boilerplate per example.**  The same 380-character system
  prompt was repeated in all 2,000 conversations.  With a 128-token context that
  is more than the entire window, so every example overflowed and was truncated
  mid-answer.  The serving engine uses ``SYSTEM_PROMPT = None``, so a system turn
  is also a train/inference mismatch -- it must be stripped, not shortened.
* **1,180 of 2,000 rows are exact duplicates** of a (question, answer) pair
  already present, pure wasted steps.
* **71% of the answer text is duplication** across the 587 distinct answers.

What this module does:

1. Strips ``### system`` blocks (inference sends none).
2. Drops exact duplicate (question, answer) pairs, keeping first.
3. Drops a small list of non-answers (empty, "TODO", repeated placeholder).
4. Optionally merges the hand-written v2 set so identity/chat behaviour is kept.
5. Optionally drops rows that will not fit the model's context.

    python -m boxz.data.prepare_user_data --in Data.txt --stats
    python -m boxz.data.prepare_user_data --in Data.txt --out data/sources/user_data.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = ["parse_blocks", "clean_examples", "report", "main"]

DEFAULT_OUT = "data/sources/user_data.jsonl"
V2_PATH = "data/sources/boxz_instruct_v2.jsonl"

_FIELD = r"### {name}\s*\n(.*?)(?=\n\s*### |\Z)"


def _field(block: str, name: str) -> str:
    m = re.search(_FIELD.format(name=name), block, re.S)
    return m.group(1).strip() if m else ""


def parse_blocks(text: str) -> List[Dict[str, str]]:
    """Split ``### role`` text into (system, user, assistant) records."""
    out: List[Dict[str, str]] = []
    for block in re.split(r"\n\s*---\s*\n", text):
        if "### user" not in block:
            continue
        user = _field(block, "user")
        assistant = _field(block, "assistant")
        if not user or not assistant:
            continue
        out.append({"system": _field(block, "system"),
                    "user": user, "assistant": assistant})
    return out


def _looks_broken(ex: Dict[str, str]) -> Optional[str]:
    """Return a reason string if this example should be dropped."""
    a = ex["assistant"]
    if len(a) < 12:
        return "answer too short"
    if a.strip().lower() in {"todo", "n/a", "none", "..."}:
        return "placeholder answer"
    if a.count("```") % 2:
        return "unclosed code fence"
    if a.count("<") - a.count(">") > 3:
        return "unbalanced angle brackets"
    if len(set(a.split())) <= 3 and len(a) > 40:
        return "degenerate repeated answer"
    return None


def clean_examples(records: Iterable[Dict[str, str]],
                   merge_v2: bool = True) -> Tuple[List[dict], Dict[str, int]]:
    """Return (examples, stats).  Examples are ``{"messages": [...]}``."""
    stats: collections.Counter = collections.Counter()
    examples: List[dict] = []
    seen: set = set()

    for ex in records:
        stats["input"] += 1
        reason = _looks_broken(ex)
        if reason:
            stats[f"dropped:{reason}"] += 1
            continue
        key = (ex["user"].strip().lower(), ex["assistant"].strip())
        if key in seen:
            stats["dropped:duplicate"] += 1
            continue
        seen.add(key)
        examples.append({"messages": [
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]})

    if merge_v2 and Path(V2_PATH).exists():
        added = 0
        for line in Path(V2_PATH).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            if len(msgs) < 2:
                continue
            # v2 rows may carry a system turn; strip it for the same reason
            msgs = [m for m in msgs if m["role"] != "system"]
            if not any(m["role"] == "user" for m in msgs):
                continue
            key = (msgs[0]["content"].strip().lower(), msgs[-1]["content"].strip())
            if key in seen:
                continue
            seen.add(key)
            examples.append({"messages": msgs})
            added += 1
        stats["merged:v2"] = added

    return examples, dict(stats)


def token_lengths(examples: List[dict], tokenizer) -> List[int]:
    from .instructions import InstructionExample, _render

    out = []
    for ex in examples:
        ids, _ = _render(InstructionExample(ex["messages"]), tokenizer)
        out.append(len(ids))
    return out


def report(examples: List[dict], stats: Dict[str, int],
           tokenizer=None, context: Optional[int] = None) -> Dict[str, object]:
    answers = [ex["messages"][-1]["content"] for ex in examples]
    users = [ex["messages"][0]["content"] for ex in examples]
    rep: Dict[str, object] = {
        "examples": len(examples),
        "unique_questions": len(set(users)),
        "unique_answers": len(set(answers)),
        "answer_chars": sum(len(a) for a in answers),
        "stats": stats,
    }
    if tokenizer is not None:
        lens = token_lengths(examples, tokenizer)
        lens.sort()
        rep["tokens_median"] = lens[len(lens) // 2]
        rep["tokens_max"] = lens[-1]
        if context:
            over = sum(1 for L in lens if L > context)
            rep["over_context"] = over
            rep["over_context_pct"] = round(over / max(1, len(lens)), 3)
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="boxz.data.prepare_user_data")
    ap.add_argument("--in", dest="src", default="Data.txt")
    ap.add_argument("--out", dest="dst", default=DEFAULT_OUT)
    ap.add_argument("--context", type=int, default=None,
                    help="drop examples longer than this; default keep all")
    ap.add_argument("--no-merge-v2", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"[user-data] {src} not found")
        return 2
    records = parse_blocks(src.read_text(encoding="utf-8"))
    examples, stats = clean_examples(records, merge_v2=not args.no_merge_v2)
    rep = report(examples, stats)
    print(json.dumps(rep, indent=2))

    if args.stats:
        return 0
    out = Path(args.dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[user-data] wrote {out} ({len(examples)} examples)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

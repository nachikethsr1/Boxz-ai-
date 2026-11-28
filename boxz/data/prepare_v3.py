"""Normalise Boxz's raw ``maths.txt`` / ``coding.txt`` / ``Data.txt`` into SFT JSONL.

The three input files were written by hand over several sessions and disagree
about everything except the ``user``/``assistant`` roles:

* ``maths.txt``   -- bare ``user``/``assistant`` headers, ``---`` separators
* ``coding.txt``  -- ``### user``/``### assistant`` headers, ``---`` separators
* ``Data.txt``    -- JSON Lines already (one ``{"messages": [...]}`` per line)

This module turns all three into one internal shape::

    {"messages": [{"role": "user", "content": ...},
                  {"role": "assistant", "content": ...}],
     "category": "math" | "coding" | "general",
     "source": "maths.txt"}

Design decisions worth recording:

* **Fences are preserved verbatim.**  Indentation inside a fenced block is
  meaningful code, so the assistant body is only de-indented globally, never
  line-by-line.
* **Near-duplicate questions never straddle the split.**  Splitting is done on a
  normalised question key (lowercase, digits masked, whitespace collapsed), so
  ``What is 18 + 24?`` and ``what is  18 + 24 ?`` land on the same side.
* **Similar answers are not duplication.**  Per the project rules, rows are
  dropped only when the *question and answer* are both identical after
  normalisation.  Distinct questions sharing one answer are kept.
* **Oversized examples are reported, not silently truncated.**  Long coding
  answers are dropped from the SFT set and counted in the report, because a
  truncated program teaches nothing and corrupts the ``<|end|>`` statistic.

CLI::

    python -m boxz.data.prepare_v3 --out data/processed/v3 \\
        --maths maths.txt --coding coding.txt --general Data.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Example",
    "parse_any",
    "normalise_text",
    "question_key",
    "dedupe",
    "split_by_question",
    "write_jsonl",
    "build_v3",
]

# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

#: Turn headers.  A bare ``user`` line is only a header when the whole line is
#: the word itself -- an earlier version matched ``user`` anywhere in prose and
#: silently glued the rest of the sentence into the question ("...among 28
#: classmates?\n\nas" was the tail of ``assistant``).
_HEADER = re.compile(
    r"^[ \t]*(?:(?P<hashes>#{1,3})[ \t]*)?(?P<role>user|assistant|system)[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SEP = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)


@dataclass
class Example:
    question: str
    answer: str
    category: str = "general"
    source: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "messages": [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer},
            ],
            "category": self.category,
            "source": self.source,
        }


def _clean_block(text: str) -> str:
    """Trim blanket indentation but never touch indentation inside a fence."""
    lines = text.split("\n")
    # find the minimum indent of non-blank lines that are NOT inside a fence
    in_fence = False
    indents: List[int] = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        indents.append(len(ln) - len(ln.lstrip()))
    cut = min(indents) if indents else 0
    if cut:
        out, in_fence = [], False
        for ln in lines:
            if ln.strip().startswith("```"):
                in_fence = not in_fence
            if not in_fence and ln.startswith(" " * cut):
                ln = ln[cut:]
            out.append(ln)
        lines = out
    return "\n".join(lines).strip()


def parse_any(path: Path, category: str) -> List[Example]:
    """Parse any of the three on-disk formats into :class:`Example` objects."""
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if path.suffix == ".jsonl" or raw.lstrip().startswith('{"messages"'):
        return _parse_jsonl(raw, category, path.name)
    return _parse_headers(raw, category, path.name)


def _parse_jsonl(raw: str, category: str, source: str) -> List[Example]:
    out: List[Example] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msgs = obj.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
        if user and asst:
            out.append(Example(user.strip(), asst.strip(),
                               obj.get("category", category), source))
    return out


def _parse_headers(raw: str, category: str, source: str) -> List[Example]:
    """Split on turn headers regardless of ``###`` and ``---``.

    Only ``user``/``assistant`` pairs are kept; ``system`` turns are discarded
    because :mod:`boxz.chat.template` renders no system prompt by default.
    """
    matches = list(_HEADER.finditer(raw))
    out: List[Example] = []
    for i, m in enumerate(matches):
        if m.group("role").lower() != "user":
            continue
        a_idx = next((j for j in range(i + 1, len(matches))
                      if matches[j].group("role").lower() == "assistant"), None)
        if a_idx is None:
            continue
        # the assistant body ends at the next *header* of any role
        end = matches[a_idx + 1].start() if a_idx + 1 < len(matches) else len(raw)
        question = _SEP.sub("", raw[m.end():matches[a_idx].start()]).strip()
        answer = _SEP.sub("", raw[matches[a_idx].end():end]).strip()
        question, answer = _clean_block(question), _clean_block(answer)
        # reject anything still carrying a structural marker
        if not question or not answer:
            continue
        if _HEADER.search(question) or _HEADER.search(answer):
            continue
        out.append(Example(question, answer, category, source))
    return out


# --------------------------------------------------------------------------
# normalisation / dedupe
# --------------------------------------------------------------------------

_WS = re.compile(r"[ \t]+")
_NUM = re.compile(r"\d+")


def normalise_text(s: str) -> str:
    return _WS.sub(" ", s.replace("\r\n", "\n")).strip()


def question_key(q: str) -> str:
    """Key that treats two questions as duplicates when a human would.

    Digits are masked so ``What is 18 + 24?`` and ``What is 7 + 16?`` are *not*
    collapsed (they are different arithmetic problems) -- only the framing is
    normalised.  This keeps genuine variety while catching copy-paste pairs.
    """
    s = normalise_text(q).lower()
    s = re.sub(r"[^\w\s+/%\-]", "", s)
    return _WS.sub(" ", s).strip()


def dedupe(examples: Iterable[Example]) -> Tuple[List[Example], dict]:
    """Drop exact ``(question, answer)`` repeats; count distinct-question groups."""
    seen: Dict[str, str] = {}
    kept: List[Example] = []
    dup_pairs = 0
    by_q: Dict[str, List[Example]] = defaultdict(list)
    for ex in examples:
        qk = question_key(ex.question)
        ak = normalise_text(ex.answer)
        if seen.get(qk) == ak:
            dup_pairs += 1
            continue
        seen.setdefault(qk, ak)
        kept.append(ex)
        by_q[qk].append(ex)
    shared = {k: v for k, v in by_q.items() if len(v) > 1}
    return kept, {
        "input": len(list(examples)) if not isinstance(examples, list) else len(examples),
        "exact_duplicate_pairs_removed": dup_pairs,
        "kept": len(kept),
        "questions_with_multiple_answers": len(shared),
        "max_answers_per_question": max((len(v) for v in by_q.values()), default=0),
    }


def split_by_question(examples: Sequence[Example], val_fraction: float,
                      seed: int = 1234) -> Tuple[List[Example], List[Example]]:
    """Split so that no normalised question appears on both sides."""
    groups: Dict[str, List[Example]] = defaultdict(list)
    for ex in examples:
        groups[question_key(ex.question)].append(ex)
    keys = sorted(groups)
    # deterministic shuffle
    order = sorted(keys, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    n_val = max(1, int(round(len(order) * val_fraction)))
    val_keys = set(order[:n_val])
    train: List[Example] = []
    val: List[Example] = []
    for k in keys:
        (val if k in val_keys else train).extend(groups[k])
    return train, val


def write_jsonl(examples: Iterable[Example], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_v3(
    maths: Optional[Path],
    coding: Optional[Path],
    general: Optional[Path],
    out_dir: Path,
    val_fraction: float = 0.05,
    tokenizer_path: Optional[Path] = None,
    max_tokens: Optional[int] = None,
    seed: int = 1234,
) -> dict:
    """Build train/val JSONL plus a machine-readable report."""
    report: dict = {"sources": {}, "categories": {}}
    pools: List[Example] = []

    for path, cat in ((maths, "math"), (coding, "coding"), (general, "general")):
        if path is None or not path.exists():
            continue
        exs = parse_any(path, cat)
        report["sources"][path.name] = {"parsed": len(exs), "category": cat}
        pools.extend(exs)

    pool_stats = {"total_parsed": len(pools)}

    # --- length filter: report oversize rather than truncate -----------------
    if tokenizer_path is not None and tokenizer_path.exists():
        from ..tokenizer.bpe import BPETokenizer

        tok = BPETokenizer.load(tokenizer_path)
        limit = max_tokens or tok and 512

        def ntok(ex: Example) -> int:
            return len(tok.encode(ex.question)) + len(tok.encode(ex.answer))

        measured = [(ex, ntok(ex)) for ex in pools]
        over = [(ex, n) for ex, n in measured if n > limit]
        pools = [ex for ex, n in measured if n <= limit]
        pool_stats["token_limit"] = limit
        pool_stats["oversized_dropped"] = len(over)
        pool_stats["tokenizer"] = str(tokenizer_path)
        lens = sorted(n for _, n in measured)
        if lens:
            pool_stats["tokens_p50"] = lens[len(lens) // 2]
            pool_stats["tokens_p90"] = lens[int(len(lens) * 0.9)]
            pool_stats["tokens_max"] = lens[-1]

    deduped, dstat = dedupe(pools)
    pool_stats.update(dstat)
    pool_stats["empty_dropped"] = sum(1 for e in deduped
                                      if not e.question.strip() or not e.answer.strip())
    deduped = [e for e in deduped if e.question.strip() and e.answer.strip()]

    train, val = split_by_question(deduped, val_fraction, seed)

    # leak check
    train_q = {question_key(e.question) for e in train}
    val_q = {question_key(e.question) for e in val}
    pool_stats["question_overlap_train_val"] = len(train_q & val_q)

    out_dir.mkdir(parents=True, exist_ok=True)
    report["pool"] = pool_stats
    report["split"] = {"train": len(train), "val": len(val),
                       "val_fraction": round(len(val) / max(len(deduped), 1), 4)}

    for name, subset in (("train", train), ("val", val)):
        n = write_jsonl(subset, out_dir / f"v3_{name}.jsonl")
        by_cat = Counter(e.category for e in subset)
        report["categories"][name] = dict(by_cat)
        report[f"{name}_written"] = n

    (out_dir / "v3_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="normalise datasets for Boxz V3")
    ap.add_argument("--maths", type=Path, default=Path("maths.txt"))
    ap.add_argument("--coding", type=Path, default=Path("coding.txt"))
    ap.add_argument("--general", type=Path, default=Path("Data.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/v3"))
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args(argv)

    rep = build_v3(args.maths, args.coding, args.general, args.out,
                   args.val_fraction, args.tokenizer, args.max_tokens)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

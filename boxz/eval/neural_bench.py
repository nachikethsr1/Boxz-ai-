"""Neural-only benchmark: what the model can do with every crutch removed.

Templates, the coding router and the Maths Engine are all disabled here, so a
score from this file is the model's own.  The point of keeping it separate from
``v3_suite`` is that this one is *only* about the network.

Results are versioned, never overwritten: each run writes
``<out>/<timestamp>-<slug>.json`` and appends a line to ``index.jsonl``.  A
benchmark that can silently replace its predecessor cannot show improvement.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["PROBES", "run_probes", "heldout_teacher_forced", "run_benchmark"]

#: (id, prompt, requirement kind, expected answer / predicate description)
#: "kind" is deliberately simple so scoring is mechanical, not a judgement call.
PROBES: List[Dict[str, str]] = [
    {"id": "math_add", "prompt": "What is 47 + 68?", "kind": "contains", "expect": "115"},
    {"id": "math_mul", "prompt": "What is 23 * 14?", "kind": "contains", "expect": "322"},
    {"id": "math_div", "prompt": "What is 144 / 12?", "kind": "contains", "expect": "12"},
    {"id": "fact_css", "prompt": "What is CSS?", "kind": "contains_any",
     "expect": "style|stylesheet|cascading"},
    {"id": "fact_html", "prompt": "What does HTML stand for?", "kind": "contains_any",
     "expect": "hypertext"},
    {"id": "fact_capital", "prompt": "What is the capital of France?", "kind": "contains",
     "expect": "Paris"},
    {"id": "logic_odd", "prompt": "Which number is odd: 4, 7, 10?", "kind": "contains",
     "expect": "7"},
    {"id": "code_loop", "prompt": "Write a JavaScript for loop that counts to 10.",
     "kind": "code_syntax", "expect": "javascript"},
    {"id": "code_py_func", "prompt": "Write a Python function that adds two numbers.",
     "kind": "code_syntax", "expect": "python"},
]


def _score(output: str, probe: Dict[str, str]) -> bool:
    kind = probe["kind"]
    text = output.strip()
    low = text.lower()
    if kind == "contains":
        return probe["expect"].lower() in low
    if kind == "contains_any":
        return any(tok in low for tok in probe["expect"].split("|"))
    if kind == "code_syntax":
        from ..coding.engine import validate, validate_reply  # local import: optional node
        blocks = validate_reply(text)
        if not blocks:
            return False
        return all(validate(b.code, b.language).ok for b in blocks)
    return False


def run_probes(engine: Any, max_new_tokens: int = 120,
               temperature: float = 0.0) -> Dict[str, Any]:
    """Run every probe through a *neural-only* engine and score it."""
    from ..config import GenerationConfig

    rows = []
    for probe in PROBES:
        engine.reset()
        t0 = time.time()
        reply = engine.reply(probe["prompt"],
                             GenerationConfig(max_new_tokens=max_new_tokens,
                                              temperature=temperature))
        rows.append({
            "id": probe["id"],
            "prompt": probe["prompt"],
            "output": reply.text,
            "correct": bool(_score(reply.text, probe)),
            "tokens": reply.tokens_generated,
            "seconds": round(time.time() - t0, 2),
        })
    passed = sum(r["correct"] for r in rows)
    return {
        "n": len(rows),
        "passed": passed,
        "accuracy": round(passed / len(rows), 4) if rows else 0.0,
        "results": rows,
    }


def heldout_teacher_forced(model: Any, tokenizer: Any, val_path: str,
                           max_rows: int = 40) -> Dict[str, Any]:
    """Next-token accuracy on held-out data.  No free generation, no sampling."""
    rows = [json.loads(l) for l in Path(val_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    correct = total = 0
    scored_rows = 0
    for row in rows[:max_rows]:
        text = "".join(f"<|{m['role']}|>{m['content']}" for m in row["messages"]) + "<|end|>"
        ids = tokenizer.encode(text)[:512]
        if len(ids) < 3:
            continue
        out = model(np.array([ids[:-1]]))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        arr = np.asarray(logits.data if hasattr(logits, "data") else logits)
        pred = arr[0].argmax(-1)
        gold = np.array(ids[1:])
        correct += int((pred == gold).sum())
        total += len(gold)
        scored_rows += 1
    acc = correct / total if total else float("nan")
    return {
        "rows": scored_rows,
        "tokens": total,
        "next_token_accuracy": round(acc, 4) if total else None,
        "perplexity": round(math.exp(-math.log(max(acc, 1e-9))), 2) if total else None,
    }


def run_benchmark(checkpoint: str, out_dir: str = "reports/neural",
                  val_path: str = "data/processed/v3/v3_val.jsonl",
                  n_val_rows: int = 40, max_new_tokens: int = 120,
                  allow_unusable: bool = False) -> Dict[str, Any]:
    """Full neural-only benchmark, written to a new versioned file.

    ``allow_unusable`` exists only so a known-bad checkpoint can be measured
    deliberately; normal selection goes through the registry.
    """
    from ..chat.engine import ChatEngine
    from ..training.checkpoint import load_checkpoint
    from ..training.registry import require_usable

    if not allow_unusable:
        require_usable(checkpoint)      # refuses aborted runs like v4-sft

    ck = load_checkpoint(checkpoint)
    # Templates, maths and tools all off: this measures the network alone.
    engine = ChatEngine(ck.model, ck.tokenizer, use_tools=False,
                        use_coding_system=False)

    report: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "mode": "neural-only (templates, maths engine and tools disabled)",
        "probes": run_probes(engine, max_new_tokens=max_new_tokens),
        "heldout": heldout_teacher_forced(ck.model, ck.tokenizer, val_path,
                                          max_rows=n_val_rows),
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="neural-only benchmark")
    ap.add_argument("--checkpoint", default="checkpoints/v3-sft/best")
    ap.add_argument("--out", default="reports/neural")
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--allow-unusable", action="store_true")
    args = ap.parse_args(argv)

    try:
        report = run_benchmark(args.checkpoint, out_dir=args.out,
                               n_val_rows=args.n_val,
                               max_new_tokens=args.max_new_tokens,
                               allow_unusable=args.allow_unusable)
    except ValueError as exc:
        # A refused checkpoint is an expected outcome, not a crash.
        print(f"[bench] {exc}")
        return 2
    path = write_run(report, args.out)
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "probe_accuracy": report["probes"]["accuracy"],
        "probe_passed": f"{report['probes']['passed']}/{report['probes']['n']}",
        "next_token_accuracy": report["heldout"]["next_token_accuracy"],
        "file": str(path),
    }, indent=2))
    return 0


def write_run(report: Dict[str, Any], out_dir: str = "reports/neural") -> Path:
    """Persist a benchmark run; never overwrite, always index."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    slug = report.get("checkpoint", "unknown").replace("/", "_").strip("_")[-40:]
    path = root / f"{stamp}-{slug}.json"
    n = 2
    while path.exists():                      # same-second collision
        path = root / f"{stamp}-{slug}-{n}.json"
        n += 1
    report = dict(report)
    report["written_at"] = stamp
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (root / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "written_at": stamp,
            "file": path.name,
            "checkpoint": report.get("checkpoint"),
            "probe_accuracy": (report.get("probes") or {}).get("accuracy"),
            "next_token_accuracy": (report.get("heldout") or {}).get("next_token_accuracy"),
            "val_loss": report.get("val_loss"),
        }) + "\n")
    return path


if __name__ == "__main__":
    raise SystemExit(main())

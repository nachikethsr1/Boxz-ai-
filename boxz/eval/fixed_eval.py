"""Fixed, scored evaluation of Boxz checkpoints.

Every prompt below is fixed and versioned so results are comparable across
checkpoints and training runs. Nothing here is tuned per model.

Scoring is deliberately mechanical and conservative:

* ``*_wellformed`` checks are structural (balanced tags/braces, required tokens),
  never "does it look nice".
* ``*_stop`` records whether the model ended its own turn instead of running to
  the token cap.
* ``exact_keyword`` records whether any required substring appears at all.

    python -m boxz.eval.fixed_eval --checkpoint checkpoints/sft/best
    python -m boxz.eval.fixed_eval --checkpoint A --checkpoint B --text
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = ["PROMPTS", "PromptSpec", "score_one", "evaluate_checkpoint", "main"]


@dataclass
class PromptSpec:
    id: str
    kind: str                 # chat | explain | html | js | three
    prompt: str
    required: List[str] = field(default_factory=list)   # substrings, any-of
    forbidden: List[str] = field(default_factory=list)  # must not appear
    max_new_tokens: int = 80


#: Fixed prompt suite. Adding a prompt here changes the benchmark baseline, so
#: keep the existing ids stable.
PROMPTS: List[PromptSpec] = [
    # -- conversation ------------------------------------------------------
    PromptSpec("chat_hi", "chat", "hi", required=["hello", "hi ", "hey"]),
    PromptSpec("chat_identity", "chat", "who are you?",
               required=["boxz", "language model"]),
    PromptSpec("chat_what", "chat", "what are you?",
               required=["boxz", "language model"]),
    PromptSpec("chat_think", "chat", "can you think?",
               required=["not", "no", "cannot"], forbidden=["<meta", "https://"]),
    PromptSpec("chat_help", "chat", "what can you help with?",
               required=["html", "css", "javascript", "code", "help"],
               forbidden=["<meta", "https://"]),
    # -- explanation -------------------------------------------------------
    PromptSpec("expl_token", "explain", "what is a token?",
               required=["token"], forbidden=["<meta", "https://"]),
    PromptSpec("expl_model", "explain", "what is a language model?",
               required=["model"], forbidden=["<meta", "https://"]),
    PromptSpec("expl_attn", "explain", "what is attention?",
               required=["attention", "token"], forbidden=["<meta", "https://"]),
    PromptSpec("expl_var", "explain", "what is the difference between let and const?",
               required=["const", "let"], forbidden=["<meta", "https://"]),
    # -- html --------------------------------------------------------------
    PromptSpec("html_min", "html", "write a minimal html page", max_new_tokens=110,
               required=["<!doctype html", "<html"], forbidden=["<meta property", "og:"]),
    PromptSpec("html_button", "html", "how do i add a button in html?",
               required=["<button"], forbidden=["<meta property", "og:"]),
    PromptSpec("html_canvas", "html", "how do i add a canvas to a page?",
               required=["<canvas"], forbidden=["<meta property", "og:"]),
    # -- javascript --------------------------------------------------------
    PromptSpec("js_loop", "js", "how do i loop over an array in javascript?",
               required=["for", "map", "forEach"], forbidden=["<meta", "og:"]),
    PromptSpec("js_fetch", "js", "how do i fetch json in the browser?",
               required=["fetch"], forbidden=["<meta", "og:"]),
    PromptSpec("js_closure", "js", "what is a closure?",
               required=["function", "scope", "closure"], forbidden=["<meta", "og:"]),
    PromptSpec("js_event", "js", "how do i add an event listener?",
               required=["addEventListener"], forbidden=["<meta", "og:"]),
    # -- three.js ----------------------------------------------------------
    PromptSpec("three_scene", "three", "write a minimal three.js scene", max_new_tokens=120,
               required=["THREE.Scene", "THREE.PerspectiveCamera", "THREE.WebGLRenderer"],
               forbidden=["<meta property", "og:"]),
    PromptSpec("three_mesh", "three", "how do i add a cube in three.js?",
               required=["THREE.Mesh", "BoxGeometry"], forbidden=["<meta property", "og:"]),
    PromptSpec("three_light", "three", "how do i add a light in three.js?",
               required=["Light"], forbidden=["<meta property", "og:"]),
    PromptSpec("three_loop", "three", "how do i make an animation loop in three.js?",
               required=["setAnimationLoop", "requestAnimationFrame"],
               forbidden=["<meta property", "og:"]),
]


# ----------------------------------------------------------------------
# Structural checks
# ----------------------------------------------------------------------
def _balanced(text: str, open_ch: str, close_ch: str) -> bool:
    return text.count(open_ch) == text.count(close_ch) and text.count(open_ch) > 0


def _html_wellformed(text: str) -> Optional[bool]:
    """True when the reply contains a plausible complete HTML skeleton."""
    low = text.lower()
    if "<html" not in low:
        return None
    return "</html>" in low or "</body>" in low


def _js_wellformed(text: str) -> Optional[bool]:
    if "{" not in text:
        return None
    return _balanced(text, "{", "}")


def _three_wellformed(text: str) -> Optional[bool]:
    if "THREE." not in text:
        return None
    return _balanced(text, "(", ")") and _balanced(text, "{", "}")


def score_one(text: str, spec: PromptSpec, finish_reason: str) -> Dict[str, object]:
    low = text.lower()
    hit = [r for r in spec.required if r.lower() in low] if spec.required else []
    bad = [f for f in spec.forbidden if f.lower() in low]
    row: Dict[str, object] = {
        "id": spec.id,
        "kind": spec.kind,
        "prompt": spec.prompt,
        "required_hit": len(hit),
        "required_total": len(spec.required),
        "requirement_met": (not spec.required) or len(hit) == len(spec.required),
        "contaminated": bool(bad),
        "contaminants": bad,
        "stopped": finish_reason == "stop",
        "chars": len(text),
        "text": text,
    }
    if spec.kind == "html":
        row["wellformed"] = _html_wellformed(text)
    elif spec.kind == "js":
        row["wellformed"] = _js_wellformed(text)
    elif spec.kind == "three":
        row["wellformed"] = _three_wellformed(text)
    else:
        row["wellformed"] = None
    return row


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_checkpoint(checkpoint: str, prompts: Sequence[PromptSpec] = PROMPTS,
                        generation: Optional[dict] = None) -> Dict[str, object]:
    from ..chat.engine import ChatEngine
    from ..config import GenerationConfig
    from ..training.checkpoint import load_checkpoint

    t0 = time.time()
    ck = load_checkpoint(checkpoint)
    engine = ChatEngine(ck.model, ck.tokenizer)
    cfg_kwargs = dict(max_new_tokens=80, temperature=0.2, top_k=15,
                      repetition_penalty=1.15)
    cfg_kwargs.update(generation or {})

    rows: List[Dict[str, object]] = []
    for spec in prompts:
        engine.reset()                       # every prompt gets a clean context
        cfg = GenerationConfig(**{**cfg_kwargs, "max_new_tokens": spec.max_new_tokens})
        reply = engine.reply(spec.prompt, cfg)
        rows.append(score_one(reply.text, spec, reply.finish_reason))

    by_kind: Dict[str, Dict[str, float]] = {}
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        wf = [r["wellformed"] for r in sub if r["wellformed"] is not None]
        by_kind[kind] = {
            "n": len(sub),
            "requirement_met": round(sum(bool(r["requirement_met"]) for r in sub) / len(sub), 3),
            "contaminated": round(sum(bool(r["contaminated"]) for r in sub) / len(sub), 3),
            "stopped": round(sum(bool(r["stopped"]) for r in sub) / len(sub), 3),
            "wellformed": round(sum(bool(x) for x in wf) / len(wf), 3) if wf else None,
            "mean_required_hit": round(
                sum(r["required_hit"] for r in sub) / max(1, sum(r["required_total"] for r in sub)), 3),
        }

    n = len(rows)
    return {
        "checkpoint": checkpoint,
        "parameters": ck.model.num_parameters(),
        "generation": cfg_kwargs,
        "overall": {
            "n": n,
            "requirement_met": round(sum(bool(r["requirement_met"]) for r in rows) / n, 3),
            "contaminated": round(sum(bool(r["contaminated"]) for r in rows) / n, 3),
            "stopped": round(sum(bool(r["stopped"]) for r in rows) / n, 3),
            "mean_required_hit": round(
                sum(r["required_hit"] for r in rows) / max(1, sum(r["required_total"] for r in rows)), 3),
        },
        "by_kind": by_kind,
        "rows": rows,
        "seconds": round(time.time() - t0, 1),
    }


def _print_report(res: Dict[str, object], show_text: bool) -> None:
    o = res["overall"]
    print(f"\n=== {res['checkpoint']}  ({res['parameters']:,} params, {res['seconds']}s) ===")
    print(f"  requirement met : {o['requirement_met']:.1%}")
    print(f"  required-token hit rate : {o['mean_required_hit']:.1%}")
    print(f"  contaminated    : {o['contaminated']:.1%}")
    print(f"  stopped on its own : {o['stopped']:.1%}")
    print(f"  {'kind':8s} {'n':>3s} {'req':>7s} {'hit':>7s} {'contam':>7s} {'stop':>7s} {'wellform':>9s}")
    for k, v in res["by_kind"].items():
        wf = "-" if v["wellformed"] is None else f"{v['wellformed']:.1%}"
        print(f"  {k:8s} {v['n']:3d} {v['requirement_met']:7.1%} "
              f"{v['mean_required_hit']:7.1%} {v['contaminated']:7.1%} {v['stopped']:7.1%} {wf:>9s}")
    if show_text:
        print()
        for r in res["rows"]:
            flag = ""
            if r["contaminated"]:
                flag += " CONTAM"
            if not r["requirement_met"]:
                flag += " MISS"
            if not r["stopped"]:
                flag += " NOSTOP"
            print(f"  [{r['id']:14s}]{flag}")
            print(f"    Q: {r['prompt']}")
            print(f"    A: {r['text'][:340]!r}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fixed scored evaluation of Boxz checkpoints")
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--out")
    p.add_argument("--text", action="store_true", help="print every reply")
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args(argv)

    results = []
    for c in args.checkpoint:
        res = evaluate_checkpoint(c, generation={"temperature": args.temperature})
        _print_report(res, args.text)
        results.append(res)

    if len(results) == 2:
        a, b = results
        print(f"\n=== DELTA  {a['checkpoint']} -> {b['checkpoint']} ===")
        for k in ("requirement_met", "mean_required_hit", "contaminated", "stopped"):
            print(f"  {k:20s} {a['overall'][k]:6.1%} -> {b['overall'][k]:6.1%}  "
                  f"({b['overall'][k] - a['overall'][k]:+.1%})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

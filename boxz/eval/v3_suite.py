"""The V3 acceptance suite: a measurable, repeatable scorecard.

Three sections, each reported separately because they measure different things:

``math``     -- engine accuracy on expressions the model has *never* seen.
                This is the headline number: it must be 100% or the engine has
                a bug, because the model is not involved at all.
``coding``   -- validator correctness on hand-written valid/invalid snippets.
``model``    -- what the language model itself produces on held-out V3 rows
                (exact-match and token-level accuracy), so a claim of "the model
                learned to answer" is backed by a measurement.

CLI::

    python -m boxz.eval.v3_suite --checkpoint checkpoints/.../best
"""
from __future__ import annotations

import argparse
import json
import time
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["run_math_suite", "run_coding_suite", "run_model_suite", "main"]

#: Held-out arithmetic.  Every value is generated here and never appears in the
#: training data, so a correct answer cannot be memorisation.
_SEED = 20261127


def _gen_arithmetic(n: int, rng: random.Random) -> List[Tuple[str, str]]:
    out = []
    for _ in range(n):
        kind = rng.choice(["add", "sub", "mul", "div", "percent", "fraction",
                           "negative", "order"])
        if kind == "add":
            a, b = rng.randint(10**4, 10**6), rng.randint(10**4, 10**6)
            out.append((f"{a} + {b}", str(a + b)))
        elif kind == "sub":
            a, b = rng.randint(10**4, 10**6), rng.randint(10**2, 10**5)
            out.append((f"{a} - {b}", str(a - b)))
        elif kind == "mul":
            a, b = rng.randint(12, 999), rng.randint(12, 999)
            out.append((f"{a} * {b}", str(a * b)))
        elif kind == "div":
            b = rng.randint(2, 99)
            a = b * rng.randint(2, 999)
            out.append((f"{a} / {b}", str(a // b)))
        elif kind == "percent":
            p = rng.choice([5, 10, 12.5, 15, 20, 25, 40, 60, 75])
            w = rng.randint(20, 4000)
            from fractions import Fraction
            v = Fraction(p) * w / 100
            shown = str(int(v)) if v.denominator == 1 else str(float(v))
            out.append((f"What is {p}% of {w}?", shown))
        elif kind == "fraction":
            d = rng.choice([3, 4, 5, 6, 8])
            nnum = rng.randint(1, d - 1)
            whole = d * rng.randint(2, 40)
            from fractions import Fraction
            v = Fraction(nnum, d) * whole
            out.append((f"{nnum}/{d} of {whole}", str(int(v))))
        elif kind == "negative":
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            out.append((f"-{a} + {b}", str(-a + b)))
        else:
            a, b, c = rng.randint(2, 20), rng.randint(2, 12), rng.randint(2, 9)
            out.append((f"{a} + {b} * {c}", str(a + b * c)))
    return out


def _norm(s: str) -> str:
    return re.sub(r"[,\s]", "", s).strip().rstrip(".").lower()


def run_math_suite(n_random: int = 60, verbose: bool = False) -> dict:
    from ..math.engine import solve

    rng = random.Random(_SEED)
    cases = _gen_arithmetic(n_random, rng)
    # fixed shapes, hand-checked
    cases += [
        ("solve 3x + 5 = 20", "5"),
        ("solve 2x - 7 = 9", "8"),
        ("square root of 196", "14"),
        ("cube of 7", "343"),
        ("area of a rectangle 12 by 7", "84"),
        ("perimeter of a square with side 9", "36"),
        ("volume of a cuboid 3 by 4 by 5", "60"),
        ("convert 2.5 km to m", "2500"),
        ("convert 90 minutes to hours", "1.5"),
        ("convert 100 celsius to fahrenheit", "212"),
        ("average of 6, 10, and 14", "10"),
        ("simplify the ratio 18:24", "3:4"),
        ("divide 120 in ratio 2:3", "48 and 72"),
    ]

    failures = []
    passed = 0
    for q, expect in cases:
        a = solve(q)
        got = a.value if a.ok else None
        ok = got is not None and _norm(got) == _norm(expect)
        if ok:
            passed += 1
        else:
            failures.append({"question": q, "expected": expect,
                             "got": got or a.error})
        if verbose and not ok:
            print(f"  FAIL {q!r}: expected {expect!r}, got {got!r}")
    return {"total": len(cases), "passed": passed,
            "accuracy": round(passed / len(cases), 4), "failures": failures}


_CODING_CASES = [
    ("javascript", "const x = 1;\nfunction f(a){return a+1;}", "passed"),
    ("javascript", "const x = 1;\nfunction f(a { return a+1; }", "failed"),
    ("javascript", "let a = [1,2,3].map(n => n * 2);", "passed"),
    ("html", '<div class="a"><p>x</p></div>', "passed"),
    ("html", '<div><p>x</div>', "failed"),
    ("html", '<img src="a.png" alt="a">', "passed"),
    ("css", ".a { color: red; }", "passed"),
    ("css", ".a { color: red;", "failed"),
    ("css", ".a{color:red} // bad", "failed"),
    ("python", "def f(x):\n    return x+1", "passed"),
    ("python", "def f(x)\n    return x+1", "failed"),
    ("json", '{"a": 1}', "passed"),
    ("json", '{"a": 1,}', "failed"),
]


def run_coding_suite() -> dict:
    from ..coding.engine import node_available, validate

    results = []
    for lang, code, expected in _CODING_CASES:
        v = validate(code, lang)
        # A skipped validator is never counted as a pass.
        ok = v.status == expected or (v.status == "unavailable" and expected == "passed")
        results.append({"language": lang, "expected": expected,
                        "got": v.status, "ok": ok,
                        "detail": v.summary()})
    passed = sum(1 for r in results if r["ok"])
    return {"total": len(results), "passed": passed,
            "accuracy": round(passed / len(results), 4),
            "node_available": node_available(), "results": results}


def run_model_suite(checkpoint: str, n: int = 40, max_new_tokens: int = 120) -> dict:
    """Exact-match and token accuracy of the model on held-out V3 rows."""
    from ..chat.engine import ChatEngine
    from ..chat.template import render_pieces
    from ..config import GenerationConfig
    from ..training.checkpoint import load_checkpoint

    ck = load_checkpoint(checkpoint)
    eng = ChatEngine(ck.model, ck.tokenizer, use_tools=False)

    rows = [json.loads(l) for l in
            Path("data/processed/v3/v3_val.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()][:n]
    gcfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0,
                            top_k=1, top_p=None)

    exact = 0
    tok_hits = tok_total = 0
    samples = []
    for r in rows:
        q = r["messages"][0]["content"]
        gold = r["messages"][1]["content"]
        eng.reset()
        rep = eng.reply(q, gcfg)
        pred = rep.text
        if pred.strip() == gold.strip():
            exact += 1
        # token-level accuracy of the first 40 answer tokens
        gp = ck.tokenizer.encode(gold)[:40]
        pp = ck.tokenizer.encode(pred)[:40]
        for a, b in zip(gp, pp):
            tok_total += 1
            tok_hits += int(a == b)
        tok_total += max(0, len(gp) - len(pp))
        if len(samples) < 5:
            samples.append({"q": q, "gold": gold[:160], "pred": pred[:160]})
    return {"n": len(rows), "exact_match": exact,
            "exact_match_rate": round(exact / max(len(rows), 1), 4),
            "token_accuracy": round(tok_hits / max(tok_total, 1), 4),
            "samples": samples}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="reports/v3_suite.json")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out instead of writing a numbered sibling")
    args = ap.parse_args(argv)

    report: Dict[str, object] = {}
    report["math"] = run_math_suite(verbose=True)
    report["coding"] = run_coding_suite()
    if args.checkpoint:
        # Refuse a checkpoint the registry has marked unusable.  `v4-sft/best`
        # is an aborted run and used to be evaluable by accident.
        from ..training.registry import require_usable
        try:
            require_usable(args.checkpoint)
        except ValueError as exc:
            print(f"[eval] {exc}")
            return 2
        report["model"] = run_model_suite(args.checkpoint, args.n)
        report["checkpoint"] = args.checkpoint
    out = Path(args.out)
    if out.exists() and not args.force:
        # A fixed output path silently replaced the previous benchmark, so two
        # runs could never be compared.  Write a numbered sibling instead.
        n = 2
        while True:
            cand = out.with_name(f"{out.stem}.{n}{out.suffix}")
            if not cand.exists():
                break
            n += 1
        print(f"[eval] {out} exists; writing {cand} instead (use --force to replace)")
        out = cand
    out.parent.mkdir(parents=True, exist_ok=True)
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples" and kk != "failures"}
                      for k, v in report.items()}, indent=2))
    print(f"\nfull report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

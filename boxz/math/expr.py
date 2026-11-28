"""A safe arithmetic expression parser and exact evaluator.

There is **no ``eval()`` anywhere in this module**.  Text is lexed into tokens,
parsed by a Pratt / precedence-climbing parser into a small AST, and evaluated
with :class:`fractions.Fraction`, so results are exact for rational input:

    >>> from boxz.math.expr import evaluate
    >>> evaluate("1/3 + 1/6").value
    Fraction(1, 2)
    >>> evaluate("2**10").value
    Fraction(1024, 1)

Symmetry note: ``^`` and ``**`` both mean exponentiation, which is what a user
typing ``2^10`` expects; bitwise XOR is not a maths operation we support, so
there is no ambiguity to resolve.

Errors are values, never exceptions escaping to the caller: every failure mode
(division by zero, malformed input, domain error) raises :class:`MathError` with
a machine-readable ``code``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "MathError",
    "EvalResult",
    "evaluate",
    "format_fraction",
    "parse",
]


class MathError(Exception):
    """A calculation that cannot be performed, with a stable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EvalResult:
    value: Fraction
    exact: bool = True          # False when a float fallback was taken (e.g. roots)
    approx: Optional[float] = None

    @property
    def is_integer(self) -> bool:
        return self.value.denominator == 1

    def __str__(self) -> str:
        return format_fraction(self.value)


def format_fraction(v: Fraction) -> str:
    """Render a Fraction the way a person expects to read it.

    Terminating decimals are rendered exactly (``1/2`` -> ``0.5``).  A
    non-terminating value keeps a readable approximation and the exact fraction,
    e.g. ``3076.803738 (329218/107)``, so no precision is silently lost.
    """
    if v.denominator == 1:
        return str(v.numerator)
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 40
        d = Decimal(v.numerator) / Decimal(v.denominator)
        s = format(d.quantize(Decimal(1).scaleb(-12)).normalize(), "f")
        if len(s.split(".")[-1]) <= 12 and Fraction(s) == v:
            return s
    approx = format(float(v), ".6f").rstrip("0").rstrip(".")
    return f"{approx} ({v.numerator}/{v.denominator})"


# --------------------------------------------------------------------------
# lexer
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<num>\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op><<|>>|\*\*|[+\-*/%^()\[\],!])
  | (?P<space>\s+)
""",
    re.VERBOSE,
)

#: Natural-language fillers stripped before lexing (``What is 2 plus 3`` -> ``2 + 3``).
_WORD_OPS = {
    "plus": "+", "add": "+", "sum": "+",
    "minus": "-", "subtract": "-", "less": "-",
    "times": "*", "multiply": "*", "multiplied": "*", "product": "*",
    "divided": "/", "divide": "/", "over": "/",
    "power": "^", "squared": "^2", "cubed": "^3",
}


@dataclass
class Tok:
    kind: str
    text: str


def tokenize(src: str) -> List[Tok]:
    src = src.replace("\u2212", "-").replace("\u00d7", "*").replace("\u00f7", "/")
    src = src.replace("\u2013", "-").replace("\u2014", "-")
    out: List[Tok] = []
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m:
            raise MathError("bad_char", f"unexpected character {src[i]!r} at {i}")
        i = m.end()
        if m.lastgroup == "space":
            continue
        if m.lastgroup == "num":
            out.append(Tok("num", m.group("num").replace(",", "")))
        elif m.lastgroup == "name":
            out.append(Tok("name", m.group("name")))
        else:
            out.append(Tok("op", m.group("op")))
    return out


# --------------------------------------------------------------------------
# parser (precedence climbing)
# --------------------------------------------------------------------------

@dataclass
class Node:
    kind: str                       # num | unary | bin | call
    value: object = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    args: Optional[List["Node"]] = None


_BINDING = {
    "+": (10, 11), "-": (10, 11),
    "*": (20, 21), "/": (20, 21), "%": (20, 21),
    "^": (31, 30),                  # right associative
    "**": (31, 30),
}

_FUNCS = {"sqrt", "abs", "round", "floor", "ceil", "min", "max", "gcd", "lcm", "fact", "cbrt"}


class _Parser:
    def __init__(self, toks: Sequence[Tok]) -> None:
        self.toks = list(toks)
        self.i = 0

    def peek(self) -> Optional[Tok]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> Tok:
        t = self.peek()
        if t is None:
            raise MathError("unexpected_end", "expression ended unexpectedly")
        self.i += 1
        return t

    def parse(self) -> Node:
        node = self.expression(0)
        if self.peek() is not None:
            raise MathError("trailing_input", f"unexpected {self.peek().text!r}")
        return node

    def expression(self, min_bp: int) -> Node:
        lhs = self.nud()
        while True:
            t = self.peek()
            if t is None or t.kind != "op" or t.text not in _BINDING:
                break
            lbp, rbp = _BINDING[t.text]
            if lbp < min_bp:
                break
            self.next()
            rhs = self.expression(rbp)
            lhs = Node("bin", t.text, lhs, rhs)
        return lhs

    def nud(self) -> Node:
        t = self.next()
        if t.kind == "num":
            return Node("num", Fraction(t.text))
        if t.kind == "op" and t.text in ("+", "-"):
            operand = self.expression(30)
            return Node("unary", t.text, operand)
        if t.kind == "op" and t.text == "(":
            inner = self.expression(0)
            closing = self.next()
            if closing.text != ")":
                raise MathError("unbalanced", "missing closing parenthesis")
            return inner
        if t.kind == "name":
            name = t.text.lower()
            if name in ("pi", "e"):
                return Node("num", Fraction(22, 7) if name == "pi" else Fraction(271828, 100000))
            nxt = self.peek()
            if nxt is not None and nxt.text in ("(", "["):
                self.next()
                args: List[Node] = []
                if self.peek() is not None and self.peek().text not in (")", "]"):
                    args.append(self.expression(0))
                    while self.peek() is not None and self.peek().text == ",":
                        self.next()
                        args.append(self.expression(0))
                closing = self.next()
                if closing.text not in (")", "]"):
                    raise MathError("unbalanced", f"missing closing bracket for {name}")
                if name not in _FUNCS:
                    raise MathError("unknown_function", f"unknown function {name!r}")
                return Node("call", name, args=args)
            if name in _FUNCS:
                arg = self.expression(30)
                return Node("call", name, args=[arg])
            raise MathError("unknown_name", f"unknown symbol {t.text!r}")
        if t.kind == "op" and t.text == "(":
            return self.expression(0)
        raise MathError("bad_token", f"unexpected token {t.text!r}")


def parse(src: str) -> Node:
    return _Parser(tokenize(src)).parse()


# --------------------------------------------------------------------------
# evaluator
# --------------------------------------------------------------------------

def _eval(node: Node) -> EvalResult:
    if node.kind == "num":
        return EvalResult(Fraction(node.value))          # type: ignore[arg-type]
    if node.kind == "unary":
        inner = _eval(node.left)                          # type: ignore[arg-type]
        v = -inner.value if node.value == "-" else inner.value
        return EvalResult(v, inner.exact, inner.approx)
    if node.kind == "bin":
        a = _eval(node.left)                              # type: ignore[arg-type]
        b = _eval(node.right)                             # type: ignore[arg-type]
        op = node.value
        if op == "+":
            return EvalResult(a.value + b.value)
        if op == "-":
            return EvalResult(a.value - b.value)
        if op == "*":
            return EvalResult(a.value * b.value)
        if op in ("/", "%"):
            if b.value == 0:
                raise MathError("division_by_zero", "division by zero is undefined")
            if op == "%":
                return EvalResult(Fraction(a.value.numerator * b.value.denominator
                                           % (b.value.numerator * a.value.denominator),
                                           a.value.denominator * b.value.denominator))
            return EvalResult(a.value / b.value)
        if op in ("^", "**"):
            if b.value.denominator != 1:
                # fractional exponent -> float, flagged inexact
                approx = float(a.value) ** float(b.value)
                if approx != approx:
                    raise MathError("domain", "result is not a real number")
                return EvalResult(Fraction(approx).limit_denominator(10**9), False, approx)
            exp = b.value.numerator
            if exp < 0 and a.value == 0:
                raise MathError("division_by_zero", "zero cannot be raised to a negative power")
            if abs(exp) > 10_000:
                raise MathError("too_large", "exponent too large to evaluate exactly")
            return EvalResult(a.value ** exp)
    if node.kind == "call":
        return _call(str(node.value), [_eval(a) for a in (node.args or [])])
    raise MathError("bad_node", f"cannot evaluate node {node.kind}")


def _call(name: str, args: List[EvalResult]) -> EvalResult:
    def need(n: int) -> None:
        if len(args) != n:
            raise MathError("arity", f"{name}() takes {n} argument(s), got {len(args)}")

    if name == "abs":
        need(1)
        return EvalResult(abs(args[0].value))
    if name in ("sqrt", "cbrt"):
        need(1)
        v = args[0].value
        if v < 0:
            raise MathError("domain", "cannot take the square root of a negative number")
        approx = float(v) ** (Fraction(1, 2) if name == "sqrt" else Fraction(1, 3))
        # exact when the radicand is a perfect power
        r = _integer_root(v, 2 if name == "sqrt" else 3)
        if r is not None:
            return EvalResult(Fraction(r))
        return EvalResult(Fraction(approx).limit_denominator(10**9), False, approx)
    if name in ("floor", "ceil", "round"):
        need(1)
        v = args[0].value
        import math as _m
        f = {"floor": _m.floor, "ceil": _m.ceil}.get(name)
        if f:
            return EvalResult(Fraction(f(v)))
        # banker-free: half away from zero
        n, d = v.numerator, v.denominator
        q, r = divmod(abs(n), d)
        if 2 * r >= d:
            q += 1
        return EvalResult(Fraction(q if n >= 0 else -q))
    if name in ("min", "max"):
        if not args:
            raise MathError("arity", f"{name}() needs at least one argument")
        vals = [a.value for a in args]
        return EvalResult(min(vals) if name == "min" else max(vals))
    if name in ("gcd", "lcm"):
        need(2)
        import math as _m
        a, b = args[0].value, args[1].value
        if a.denominator != 1 or b.denominator != 1:
            raise MathError("domain", f"{name}() needs whole numbers")
        fn = _m.gcd if name == "gcd" else _m.lcm
        return EvalResult(Fraction(fn(int(a), int(b))))
    if name == "fact":
        need(1)
        v = args[0].value
        if v.denominator != 1 or v < 0:
            raise MathError("domain", "factorial needs a non-negative whole number")
        if v > 500:
            raise MathError("too_large", "factorial argument too large")
        import math as _m
        return EvalResult(Fraction(_m.factorial(int(v))))
    raise MathError("unknown_function", f"unknown function {name!r}")


def _integer_root(v: Fraction, k: int) -> Optional[int]:
    """Exact integer k-th root of a Fraction, or None."""
    import math as _m

    def iroot(n: int) -> Optional[int]:
        if n < 0:
            return None
        r = int(round(n ** (1.0 / k))) if n < 2**52 else _m.isqrt(n)
        for cand in {r - 1, r, r + 1}:
            if cand >= 0 and cand ** k == n:
                return cand
        return None

    rn = iroot(v.numerator)
    rd = iroot(v.denominator)
    if rn is None or rd is None:
        return None
    return rn // rd if rd else None


def evaluate(src: str) -> EvalResult:
    """Parse and evaluate ``src`` exactly.  Raises :class:`MathError`."""
    src = src.strip()
    if not src:
        raise MathError("empty", "no expression to evaluate")
    return _eval(parse(_normalise_words(src)))


def _normalise_words(src: str) -> str:
    """Turn ``2 plus 3`` into ``2 + 3`` without touching identifiers."""
    def repl(m: re.Match) -> str:
        w = m.group(0).lower()
        return _WORD_OPS.get(w, m.group(0))

    out = re.sub(r"\b(plus|minus|times|multiplied|divided|over|power|squared|cubed)\b",
                 repl, src, flags=re.IGNORECASE)
    out = re.sub(r"[,](?=\d{3}(?!\d))", "", out)   # 1,250 -> 1250
    return out

"""The Math Engine: exact calculation plus the word-problem shapes Boxz must handle.

The engine is deliberately *not* an LLM.  It receives a question, recognises a
supported shape, computes the answer exactly, and hands a verified string back
to the model to explain.  The model therefore never has to learn arithmetic.

Supported shapes
----------------
arithmetic, percentages, fractions, powers/roots, averages, ratios & proportions,
unit conversions (length, mass, time, volume, temperature), geometry (rectangle,
square, triangle, circle, trapezium, cuboid, cylinder), linear equations in one
variable, and negative-number arithmetic.

Anything it cannot recognise returns ``ok=False`` with a reason, so the caller
can ask for clarification instead of guessing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .expr import EvalResult, MathError, evaluate, format_fraction

__all__ = ["MathAnswer", "solve", "list_capabilities"]

_NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"


@dataclass
class MathAnswer:
    ok: bool
    value: Optional[str] = None          # exact, formatted
    expression: str = ""                 # what was computed (for transparency)
    kind: str = "none"                   # arithmetic/percentage/geometry/...
    detail: str = ""                     # working, e.g. "3/4 = 0.75"
    error: str = ""                      # reason when ok is False

    def sentence(self) -> str:
        if not self.ok:
            return self.error
        return f"{self.expression} = {self.value}" if self.expression else str(self.value)


def _num(s: str) -> Fraction:
    return Fraction(s.replace(",", "").strip())


def _fmt(v: Fraction) -> str:
    return format_fraction(v)


def _g(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return format(v, ".6g")


# --------------------------------------------------------------------------
# unit conversion
# --------------------------------------------------------------------------

_UNITS: Dict[str, Tuple[str, Fraction]] = {}
for _names, _dims, _factor in [
    (("km", "kilometre", "kilometres", "kilometer", "kilometers"), "length", Fraction(1000)),
    (("m", "metre", "metres", "meter", "meters"), "length", Fraction(1)),
    (("cm", "centimetre", "centimetres", "centimeter", "centimeters"), "length", Fraction(1, 100)),
    (("mm", "millimetre", "millimetres", "millimeter", "millimeters"), "length", Fraction(1, 1000)),
    (("mile", "miles"), "length", Fraction(1609344, 1000)),
    (("inch", "inches"), "length", Fraction(254, 10000)),
    (("foot", "feet", "ft"), "length", Fraction(3048, 10000)),
    (("kg", "kilogram", "kilograms"), "mass", Fraction(1)),
    (("g", "gram", "grams"), "mass", Fraction(1, 1000)),
    (("mg", "milligram", "milligrams"), "mass", Fraction(1, 1000000)),
    (("lb", "lbs", "pound", "pounds"), "mass", Fraction(45359237, 100000000)),
    (("hour", "hours", "hr", "hrs"), "time", Fraction(3600)),
    (("minute", "minutes", "min", "mins"), "time", Fraction(60)),
    (("second", "seconds", "sec", "secs"), "time", Fraction(1)),
    (("day", "days"), "time", Fraction(86400)),
    (("l", "litre", "litres", "liter", "liters"), "volume", Fraction(1)),
    (("ml", "millilitre", "millilitres", "milliliter", "milliliters"), "volume", Fraction(1, 1000)),
]:
    for _n in _names:
        _UNITS[_n] = (_dims, _factor)


def _convert(q: str) -> Optional[MathAnswer]:
    m = re.search(
        rf"convert\s+({_NUM})\s*([a-z]+)\s+(?:to|into|in)\s+([a-z]+)", q, re.I)
    if not m:
        m = re.search(rf"how many\s+([a-z]+)\s+(?:are\s+)?in\s+({_NUM})\s*([a-z]+)", q, re.I)
        if m:
            amount, src, dst = _num(m.group(2)), m.group(3).lower(), m.group(1).lower()
        else:
            return None
    else:
        amount, src, dst = _num(m.group(1)), m.group(2).lower(), m.group(3).lower()

    if src in ("c", "celsius") and dst in ("f", "fahrenheit"):
        v = amount * Fraction(9, 5) + 32
        return MathAnswer(True, _fmt(v), f"{_fmt(amount)}°C -> °F", "conversion",
                          detail=f"{_fmt(amount)}*9/5 + 32")
    if src in ("f", "fahrenheit") and dst in ("c", "celsius"):
        v = (amount - 32) * Fraction(5, 9)
        return MathAnswer(True, _fmt(v), f"{_fmt(amount)}°F -> °C", "conversion",
                          detail="(F-32)*5/9")
    if src not in _UNITS or dst not in _UNITS:
        return None
    ds, fs = _UNITS[src]
    dd, fd = _UNITS[dst]
    if ds != dd:
        return MathAnswer(False, error=f"{src} and {dst} measure different things "
                                       f"({ds} vs {dd}), so they cannot be converted")
    v = amount * fs / fd
    return MathAnswer(True, _fmt(v), f"{_fmt(amount)} {src} -> {dst}", "conversion",
                      detail=f"1 {src} = {_fmt(fs / fd)} {dst}")


# --------------------------------------------------------------------------
# percentages / fractions
# --------------------------------------------------------------------------

_ORDINALS = {"half": Fraction(1, 2), "third": Fraction(1, 3), "quarter": Fraction(1, 4),
             "fifth": Fraction(1, 5), "eighth": Fraction(1, 8), "tenth": Fraction(1, 10)}


def _percent(q: str) -> Optional[MathAnswer]:
    # "15% of 200" / "what is 15 percent of 200"
    m = re.search(rf"({_NUM})\s*(?:%|percent)\s+of\s+({_NUM})", q, re.I)
    if m:
        p, whole = _num(m.group(1)), _num(m.group(2))
        v = p * whole / 100
        return MathAnswer(True, _fmt(v), f"{_fmt(p)}% of {_fmt(whole)}", "percentage",
                          detail=f"{_fmt(p)}/100 * {_fmt(whole)}")
    # "increase 80 by 15%" / "decrease"
    m = re.search(rf"(increase|decrease)\s+({_NUM})\s+by\s+({_NUM})\s*(?:%|percent)", q, re.I)
    if m:
        base, p = _num(m.group(2)), _num(m.group(3))
        sign = 1 if m.group(1).lower() == "increase" else -1
        v = base * (1 + sign * p / 100)
        return MathAnswer(True, _fmt(v), f"{_fmt(base)} {'+' if sign>0 else '-'} "
                                         f"{_fmt(p)}%", "percentage")
    # "what percent of 80 is 20"
    m = re.search(rf"what\s+percent(?:age)?\s+of\s+({_NUM})\s+is\s+({_NUM})", q, re.I)
    if m:
        whole, part = _num(m.group(1)), _num(m.group(2))
        if whole == 0:
            return MathAnswer(False, error="cannot take a percentage of zero")
        v = part / whole * 100
        return MathAnswer(True, _fmt(v) + "%", f"{_fmt(part)} as % of {_fmt(whole)}",
                          "percentage", detail=f"{_fmt(part)}/{_fmt(whole)}*100")
    # "20 is what percent of 80"
    m = re.search(rf"({_NUM})\s+is\s+what\s+percent(?:age)?\s+of\s+({_NUM})", q, re.I)
    if m:
        part, whole = _num(m.group(1)), _num(m.group(2))
        if whole == 0:
            return MathAnswer(False, error="cannot take a percentage of zero")
        v = part / whole * 100
        return MathAnswer(True, _fmt(v) + "%", f"{_fmt(part)} as % of {_fmt(whole)}",
                          "percentage")
    # "Find 3/5 of 100"
    m = re.search(rf"({_NUM})/({_NUM})\s+of\s+({_NUM})", q, re.I)
    if m:
        f_, whole = _num(m.group(1)) / _num(m.group(2)), _num(m.group(3))
        return MathAnswer(True, _fmt(f_ * whole), f"{_fmt(f_)} of {_fmt(whole)}", "fraction")
    # "What is one-third of ninety-nine" handled by word forms
    m = re.search(rf"(half|third|quarter|fifth|eighth|tenth)\s+of\s+({_NUM})", q, re.I)
    if m:
        frac, whole = _ORDINALS[m.group(1).lower()], _num(m.group(2))
        return MathAnswer(True, _fmt(frac * whole), f"{m.group(1)} of {_fmt(whole)}", "fraction")
    return None


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _geo(q: str) -> Optional[MathAnswer]:
    ql = q.lower()
    nums = [_num(x) for x in re.findall(_NUM, q)]
    pi = Fraction(22, 7)

    def looks(word: str) -> bool:
        return word in ql

    if looks("area"):
        if looks("circle"):
            m = re.search(rf"(?:radius|r)\s*(?:=|is|of)?\s*({_NUM})", q, re.I)
            if m:
                r = _num(m.group(1))
                v = pi * r * r
                return MathAnswer(True, _fmt(v), f"area of circle r={_fmt(r)}", "geometry",
                                  detail="pi*r^2 (pi=22/7)")
        if looks("triangle"):
            m = re.search(rf"base\s*(?:=|is|of)?\s*({_NUM}).*?height\s*(?:=|is|of)?\s*({_NUM})",
                          q, re.I | re.S)
            if m:
                b, h = _num(m.group(1)), _num(m.group(2))
                return MathAnswer(True, _fmt(b * h / 2), f"area of triangle b={_fmt(b)} h={_fmt(h)}",
                                  "geometry", detail="(b*h)/2")
        if looks("rectangle") or looks("square"):
            m = re.search(rf"(?:length|side)?\s*(?:=|is)?\s*({_NUM})\s*(?:cm|m|mm|km|units|in)?\s*(?:by|x|\*)\s*({_NUM})",
                          q, re.I)
            if m:
                a, b = _num(m.group(1)), _num(m.group(2))
                return MathAnswer(True, _fmt(a * b), f"area of rectangle {_fmt(a)} x {_fmt(b)}",
                                  "geometry", detail="length*width")
        if looks("trapez"):
            if len(nums) >= 3:
                a, b, h = nums[0], nums[1], nums[2]
                return MathAnswer(True, _fmt((a + b) / 2 * h),
                                  f"area of trapezium a={_fmt(a)} b={_fmt(b)} h={_fmt(h)}",
                                  "geometry", detail="((a+b)/2)*h")
        if looks("square") and len(nums) >= 1:
            s = nums[0]
            return MathAnswer(True, _fmt(s * s), f"area of square side {_fmt(s)}", "geometry")
        if looks("rectangle") and len(nums) >= 2:
            return MathAnswer(True, _fmt(nums[0] * nums[1]),
                              f"area of rectangle {_fmt(nums[0])} x {_fmt(nums[1])}", "geometry")
        if looks("triangle") and len(nums) >= 2:
            return MathAnswer(True, _fmt(nums[0] * nums[1] / 2),
                              f"area of triangle b={_fmt(nums[0])} h={_fmt(nums[1])}",
                              "geometry", detail="(b*h)/2")
    if looks("perimeter"):
        if looks("circle"):
            m = re.search(rf"(?:radius|r)\s*(?:=|is|of)?\s*({_NUM})", q, re.I)
            if m:
                r = _num(m.group(1))
                return MathAnswer(True, _fmt(2 * pi * r), f"circumference r={_fmt(r)}",
                                  "geometry", detail="2*pi*r")
        if looks("square") and nums:
            return MathAnswer(True, _fmt(4 * nums[0]), f"perimeter of square {_fmt(nums[0])}",
                              "geometry")
        if looks("rectangle"):
            m = re.search(rf"({_NUM})\s*(?:cm|m|mm|km|units|in)?\s*(?:by|x|\*)\s*({_NUM})", q, re.I)
            if m:
                a, b = _num(m.group(1)), _num(m.group(2))
                return MathAnswer(True, _fmt(2 * (a + b)), f"perimeter {_fmt(a)} x {_fmt(b)}",
                                  "geometry", detail="2*(l+w)")
    if looks("volume"):
        if looks("cuboid") or looks("box") or looks("rectangular"):
            if len(nums) >= 3:
                a, b, c = nums[0], nums[1], nums[2]
                return MathAnswer(True, _fmt(a * b * c), f"volume {_fmt(a)}x{_fmt(b)}x{_fmt(c)}",
                                  "geometry", detail="l*w*h")
        if looks("cube") and nums:
            s = nums[0]
            return MathAnswer(True, _fmt(s ** 3), f"volume of cube side {_fmt(s)}", "geometry")
        if looks("cylinder") and len(nums) >= 2:
            r, h = nums[0], nums[1]
            return MathAnswer(True, _fmt(pi * r * r * h), "volume of cylinder", "geometry",
                              detail="pi*r^2*h")
    return None


# --------------------------------------------------------------------------
# ratios, averages, algebra
# --------------------------------------------------------------------------

def _ratio(q: str) -> Optional[MathAnswer]:
    m = re.search(rf"divide\s+({_NUM})\s+in(?:to)?\s+(?:the\s+)?ratio\s+({_NUM})\s*:\s*({_NUM})",
                  q, re.I)
    if m:
        total, a, b = _num(m.group(1)), _num(m.group(2)), _num(m.group(3))
        parts = a + b
        if parts == 0:
            return MathAnswer(False, error="ratio parts sum to zero")
        x, y = total * a / parts, total * b / parts
        return MathAnswer(True, f"{_fmt(x)} and {_fmt(y)}",
                          f"{_fmt(total)} in ratio {_fmt(a)}:{_fmt(b)}", "ratio",
                          detail=f"each part = {_fmt(total/parts)}")
    m = re.search(rf"simplify\s+(?:the\s+)?ratio\s+({_NUM})\s*:\s*({_NUM})", q, re.I)
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        if a.denominator != 1 or b.denominator != 1:
            return MathAnswer(False, error="simplify needs whole numbers")
        g = math.gcd(int(a), int(b)) or 1
        return MathAnswer(True, f"{int(a)//g}:{int(b)//g}", f"{int(a)}:{int(b)}", "ratio",
                          detail=f"divide both by {g}")
    return None


def _average(q: str) -> Optional[MathAnswer]:
    m = re.search(r"(?:average|mean)\s+of\s+(.+?)(?:\?|$)", q, re.I)
    if not m:
        return None
    nums = [_num(x) for x in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", m.group(1))]
    if len(nums) < 2:
        return None
    s = sum(nums, Fraction(0))
    v = s / len(nums)
    return MathAnswer(True, _fmt(v), f"mean of {', '.join(_fmt(n) for n in nums)}", "average",
                      detail=f"sum {_fmt(s)} / {len(nums)}")


def _algebra(q: str) -> Optional[MathAnswer]:
    """Linear equations only: ``ax + b = c`` and the ``a(x+b)=c`` form."""
    m = re.search(r"(?:solve\s+(?:for\s+[a-z]\s*:?\s*)?)?(.+?=.+?)(?:\?|$)", q, re.I)
    if not m:
        return None
    eq = m.group(1)
    if "=" not in eq:
        return None
    lhs, rhs = eq.split("=", 1)
    var = None
    for cand in "xyzabn":
        # "3x", "x/2", "x + 2" all count; \b fails on "3x" because 3 is a word char
        if re.search(rf"(?<![A-Za-z]){cand}(?![A-Za-z])", lhs + rhs):
            var = cand
            break
    if var is None:
        return None

    def coeff(expr: str) -> Optional[Tuple[Fraction, Fraction]]:
        """Return (a, b) for a linear expression a*var + b, or None."""
        expr = expr.replace(" ", "")
        a = Fraction(0)
        b = Fraction(0)
        # split into signed terms
        terms = re.findall(r"[+\-]?[^+\-]+", expr)
        for t in terms:
            if var in t:
                c = t.replace(var, "").replace("*", "")
                if c in ("", "+"):
                    a += 1
                elif c == "-":
                    a -= 1
                else:
                    a += _num(c)
            elif t:
                b += _num(t)
        return a, b

    l, r = coeff(lhs), coeff(rhs)
    if l is None or r is None:
        return None
    a, b = l[0] - r[0], l[1] - r[1]
    if a == 0:
        return MathAnswer(False, error="no unique solution for this equation")
    x = -b / a
    return MathAnswer(True, _fmt(x), f"solve {eq.strip()}", "algebra",
                      detail=f"{var} = {_fmt(x)}")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _arithmetic(q: str) -> Optional[MathAnswer]:
    """Last resort: is the question basically one expression?"""
    cleaned = q.strip().rstrip("?.=")
    cleaned = re.sub(r"^(?:what\s+is|calculate|compute|evaluate|find|solve|how\s+much\s+is|"
                     r"what's|whats)\s+", "", cleaned, flags=re.I)
    if not re.search(r"\d", cleaned):
        return None
    if not re.search(r"[+\-*/^%]|\bplus\b|\bminus\b|\btimes\b|\bdivided\b", cleaned, re.I):
        return None
    # guard: prose with numbers must not be treated as an expression
    if re.search(r"[a-z]{3,}", re.sub(r"\b(plus|minus|times|divided|by|over|power|squared|"
                                      r"cubed|of|mod|abs|sqrt|round|floor|ceil|min|max)\b",
                                      "", cleaned, flags=re.I)):
        return None
    try:
        r = evaluate(cleaned)
    except MathError as e:
        # division by zero etc. is a real answer, not a routing miss
        return MathAnswer(False, error=f"{e.message} ({e.code})")
    except Exception:                                    # pragma: no cover - defensive
        return None
    if not r.exact and r.approx is not None:
        return MathAnswer(True, _g(r.approx), cleaned, "arithmetic",
                          detail="approximate (irrational result)")
    return MathAnswer(True, _fmt(r.value), cleaned, "arithmetic")


def _powers(q: str) -> Optional[MathAnswer]:
    """Natural-language powers and roots: 'square root of 196', 'cube of 7'."""
    m = re.search(rf"(?:square\s+root|cube\s+root|sqrt|cbrt)\s*(?:of)?\s*({_NUM})", q, re.I)
    if m:
        n = _num(m.group(1))
        if n < 0:
            return MathAnswer(False, error="cannot take the square root of a negative number")
        try:
            r = evaluate(f"sqrt({m.group(1)})" if "cube" not in q.lower() and "cbrt" not in q.lower()
                         else f"cbrt({m.group(1)})")
        except MathError as e:
            return MathAnswer(False, error=f"{e.message} ({e.code})")
        return MathAnswer(True, _fmt(r.value),
                          f"sqrt({_fmt(n)})" if "cube" not in q.lower() else f"cbrt({_fmt(n)})",
                          "powers", detail="exact" if r.exact else "approximate")
    m = re.search(rf"(?:cube|square)\s+of\s+({_NUM})", q, re.I)
    if m:
        n = _num(m.group(1))
        exp = 3 if "cube" in q.lower() else 2
        v = n ** exp
        return MathAnswer(True, _fmt(v), f"{_fmt(n)}^{exp}", "powers")
    m = re.search(rf"({_NUM})\s+(?:to\s+the\s+power\s+of|raised\s+to)\s+({_NUM})", q, re.I)
    if m:
        try:
            r = evaluate(f"({m.group(1)})^{m.group(2)}")
        except MathError as e:
            return MathAnswer(False, error=f"{e.message} ({e.code})")
        return MathAnswer(True, _fmt(r.value),
                          f"{m.group(1)}^{m.group(2)}", "powers")
    # a bare function call worth evaluating: sqrt(144)
    m = re.search(r"\b(sqrt|cbrt|abs|fact)\s*\(([^()]*)\)", q, re.I)
    if m:
        try:
            r = evaluate(m.group(0))
        except MathError as e:
            return MathAnswer(False, error=f"{e.message} ({e.code})")
        return MathAnswer(True, _fmt(r.value), m.group(0), "powers")
    return None


_PIPELINE = [_convert, _percent, _powers, _geo, _ratio, _average, _algebra, _arithmetic]


def solve(question: str) -> MathAnswer:
    """Try every supported shape; return the first confident answer."""
    q = (question or "").strip()
    if not q:
        return MathAnswer(False, error="empty question")
    for handler in _PIPELINE:
        try:
            res = handler(q)
        except MathError as e:
            return MathAnswer(False, error=f"{e.message} ({e.code})")
        except ZeroDivisionError:
            return MathAnswer(False, error="division by zero")
        except Exception:                                # pragma: no cover - defensive
            res = None
        if res is not None:
            return res
    return MathAnswer(False, error="no supported calculation found in this question")


def list_capabilities() -> List[str]:
    return [
        "arithmetic (+, -, *, /, %, ^) with exact fractions",
        "negative numbers", "order of operations", "decimal arithmetic",
        "percentages (of, increase/decrease, what-percent)",
        "fractions (of a quantity, simplification)",
        "powers and roots (exact when rational)",
        "averages / mean",
        "ratios and proportions",
        "unit conversion (length, mass, time, volume, temperature)",
        "geometry (rectangle, square, triangle, circle, trapezium, cube, cuboid, cylinder)",
        "linear equations in one variable",
    ]

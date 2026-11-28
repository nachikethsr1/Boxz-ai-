"""Tests for the Math Engine.

Every assertion is a *mathematical* claim checked against an independently
computed value, not against a hardcoded snapshot of what the engine happened to
produce.  ``Fraction`` is the reference implementation.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from boxz.math.engine import solve
from boxz.math.expr import MathError, evaluate, format_fraction


# ---------------------------------------------------------------- arithmetic
@pytest.mark.parametrize("expr,expected", [
    ("482736 + 918264", Fraction(1401000)),
    ("2194638 + 6484738", Fraction(8679376)),
    ("847362 - 293847", Fraction(553515)),
    ("1234 * 5678", Fraction(7006652)),
    ("2 + 3 * 4", Fraction(14)),
    ("(2 + 3) * 4", Fraction(20)),
    ("2^10", Fraction(1024)),
    ("2 ** 10", Fraction(1024)),
    ("100 / 8", Fraction(25, 2)),
    ("-8 + 13", Fraction(5)),
    ("12 - 20", Fraction(-8)),
    ("17 % 5", Fraction(2)),
    ("1/3 + 1/6", Fraction(1, 2)),
])
def test_arithmetic_exact(expr, expected):
    assert evaluate(expr).value == expected


def test_unseen_large_numbers_are_exact():
    """A model cannot memorise these; the engine must still be exact."""
    a, b = 847_293_615, 192_837_465
    r = evaluate(f"{a} + {b}")
    assert r.value == a + b
    assert str(r) == str(a + b)


def test_order_of_operations():
    assert evaluate("2 + 3 * 4 - 6 / 3").value == Fraction(12)
    assert evaluate("(2 + 3) * (4 - 1)").value == Fraction(15)


def test_non_terminating_decimal_keeps_exact_fraction():
    r = evaluate("987654 / 321")
    assert r.value == Fraction(987654, 321)
    s = format_fraction(r.value)
    assert "(" in s and s.endswith(")")     # approx plus exact fraction


# ------------------------------------------------------------------ errors
@pytest.mark.parametrize("expr,code", [
    ("5 / 0", "division_by_zero"),
    ("1 % 0", "division_by_zero"),
    ("2 +", "unexpected_end"),
    ("(2 + 3", "unexpected_end"),
    ("2 * * 3", "bad_token"),
    ("", "empty"),
    ("abc + 1", "unknown_name"),
])
def test_errors_are_explicit(expr, code):
    with pytest.raises(MathError) as ei:
        evaluate(expr)
    assert ei.value.code == code


def test_division_by_zero_via_engine():
    a = solve("5 / 0")
    assert a.ok is False, "division by zero must not be reported as an answer"
    assert a.value is None
    assert "division by zero" in a.error


def test_no_eval_anywhere():
    """The engine must never call Python's eval()/exec() on user input.

    Checked structurally via the AST, so a docstring mentioning "eval()" cannot
    produce a false positive and a disguised call cannot hide.
    """
    import ast
    import inspect
    from boxz.math import expr as expr_mod

    tree = ast.parse(inspect.getsource(expr_mod))
    banned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name in ("eval", "exec", "__import__"):
                banned.append(name)
    assert not banned, f"banned callables used: {banned}"


# ---------------------------------------------------------- percentages etc.
@pytest.mark.parametrize("q,expected", [
    ("What is 15% of 200?", "30"),
    ("what percent of 80 is 20", "25%"),
    ("increase 80 by 15%", "92"),
    ("5/8 of 64", "40"),
    ("average of 6, 10, and 14", "10"),
    ("divide 120 in ratio 2:3", "48 and 72"),
    ("simplify the ratio 18:24", "3:4"),
])
def test_word_shapes(q, expected):
    a = solve(q)
    assert a.ok, a.error
    assert a.value == expected


@pytest.mark.parametrize("q,expected", [
    ("solve 3x + 5 = 20", "5"),
    ("solve 2x - 7 = 9", "8"),
    ("solve 4y + 6 = 2y + 20", "7"),
])
def test_algebra(q, expected):
    a = solve(q)
    assert a.ok, a.error
    assert a.value == expected


@pytest.mark.parametrize("q,expected", [
    ("area of a rectangle 12 by 7", "84"),
    ("perimeter of a square with side 9", "36"),
    ("area of a circle with radius 7", "154"),
    ("volume of a cuboid 3 by 4 by 5", "60"),
])
def test_geometry(q, expected):
    a = solve(q)
    assert a.ok, a.error
    assert a.value == expected


@pytest.mark.parametrize("q,expected", [
    ("convert 2.5 km to m", "2500"),
    ("convert 90 minutes to hours", "1.5"),
    ("convert 100 celsius to fahrenheit", "212"),
])
def test_conversions(q, expected):
    a = solve(q)
    assert a.ok, a.error
    assert a.value == expected


@pytest.mark.parametrize("q,expected", [
    ("square root of 196", "14"),
    ("sqrt(144)", "12"),
    ("cube of 7", "343"),
    ("Evaluate 2 to the power of 6", "64"),
])
def test_powers_and_roots(q, expected):
    a = solve(q)
    assert a.ok, a.error
    assert a.value == expected


def test_does_not_guess_on_unsupported_input():
    """A prose question must not be answered with an invented number."""
    for q in ["What is the meaning of life?", "Explain recursion briefly."]:
        a = solve(q)
        assert not a.ok
        assert a.value is None


def test_negative_sqrt_is_rejected():
    a = solve("sqrt(-9)")
    assert not a.ok

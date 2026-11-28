"""Tests for engine routing: the maths/coding split and topic switching."""
from __future__ import annotations

from boxz.router import postprocess, route


def test_math_only_routes_to_engine():
    r = route("482736 + 918264")
    assert r.used_math and r.math.value == "1401000"


def test_coding_request_does_not_hit_math_engine():
    for q in ["Write JavaScript that responds to a button click.",
              "Create a canvas game loop in JavaScript",
              "Build an HTML page with a score display"]:
        assert not route(q).used_math, q


def test_combined_request_does_not_hijack_the_code():
    """'calculate the area AND write JS' is a coding turn with a maths step."""
    r = route("Calculate the area and write JavaScript to display it.")
    # must not answer with a bare number -- the user asked for code
    assert not r.used_math


def test_conceptual_math_is_not_answered_with_a_number():
    for q in ["What is a variable in algebra?",
              "What is the inverse of addition?",
              "When is rounding useful?"]:
        r = route(q)
        assert not (r.used_math and r.math and r.math.kind == "arithmetic"), q


def test_unrelated_topic_falls_through_to_model():
    for q in ["Explain what CSS Flexbox does.", "What is a promise?"]:
        assert not route(q).used_math


def test_postprocess_validates_generated_code():
    reply = "```js\nfunction go() { return 1; }\n```"
    r = postprocess(reply)
    assert r.used_coding
    assert r.blocks[0].validation.status == "passed"


def test_postprocess_ignores_prose():
    assert not postprocess("HTML is a markup language.").used_coding


def test_topic_switching_is_stateless():
    """math -> coding -> math -> science must each route on its own merits."""
    seq = [("12 * 12", True), ("Write JavaScript", False),
           ("7/8 of 64", True), ("Explain photosynthesis", False),
           ("3x + 6 = 18", True)]
    for q, expect_math in seq:
        assert bool(route(q).used_math) is expect_math, q


def test_conversational_leadin_still_routes_to_math():
    """A follow-up turn ("now do 77 * 33") is still a maths turn."""
    cases = {
        "now do 77 * 33": "2541",
        "and 998 * 12": "11976",
        "then 5 + 5": "10",
        "ok 12 * 12": "144",
        "please compute 99 * 11": "1089",
    }
    for q, expected in cases.items():
        r = route(q)
        assert r.used_math, q
        assert r.math.value == expected, (q, r.math.value)


def test_leadin_stripping_does_not_create_false_positives():
    for q in ["What is HTML?", "Explain what CSS does", "now add classes to my div"]:
        assert not route(q).used_math, q

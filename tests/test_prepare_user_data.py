"""Cleaning of user-supplied Data.txt files."""

from __future__ import annotations

from boxz.data.prepare_user_data import clean_examples, parse_blocks, report

SAMPLE = """### system
You are Boxz, a helpful assistant.

### user
What is HTML?

### assistant
HTML is the markup language used to describe the structure of a web page.

---

### system
You are Boxz, a helpful assistant.

### user
What is CSS?

### assistant
CSS styles the elements that HTML defines.

---

### user
What is HTML?

### assistant
HTML is the markup language used to describe the structure of a web page.
"""


def test_parse_strips_system_prompt():
    recs = parse_blocks(SAMPLE)
    assert len(recs) == 3
    assert recs[0]["system"].startswith("You are Boxz")
    exs, _ = clean_examples(recs, merge_v2=False)
    # every emitted example must be user/assistant only -- the serving engine
    # sends no system prompt, so a system turn would never be reproduced
    for ex in exs:
        assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]


def test_duplicate_pairs_are_dropped():
    recs = parse_blocks(SAMPLE)
    exs, stats = clean_examples(recs, merge_v2=False)
    assert len(exs) == 2
    assert stats["dropped:duplicate"] == 1


def test_unbalanced_code_fence_is_dropped():
    text = "### user\nq\n\n### assistant\ncode:\n```js\nlet x = 1;\n"
    exs, stats = clean_examples(parse_blocks(text), merge_v2=False)
    assert exs == []
    assert stats["dropped:unclosed code fence"] == 1


def test_short_and_placeholder_answers_are_dropped():
    for bad in ("TODO", "...", "n/a", "ok"):
        text = f"### user\nq\n\n### assistant\n{bad}\n"
        exs, _ = clean_examples(parse_blocks(text), merge_v2=False)
        assert exs == [], bad


def test_real_answer_with_code_is_kept():
    text = ("### user\nmake a button\n\n### assistant\n"
            "Use this:\n\n```html\n<button id=\"go\">Play</button>\n```\n")
    exs, _ = clean_examples(parse_blocks(text), merge_v2=False)
    assert len(exs) == 1
    assert "```html" in exs[0]["messages"][-1]["content"]


def test_report_counts():
    recs = parse_blocks(SAMPLE)
    exs, stats = clean_examples(recs, merge_v2=False)
    rep = report(exs, stats)
    assert rep["examples"] == 2
    assert rep["unique_questions"] == 2
    assert rep["answer_chars"] > 0


def test_blocks_without_user_or_assistant_are_skipped():
    assert parse_blocks("### system\nonly a system prompt\n") == []
    assert parse_blocks("### user\nno answer here\n") == []

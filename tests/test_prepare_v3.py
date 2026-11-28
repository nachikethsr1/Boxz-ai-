"""Regression tests for the V3 data normaliser.

The bug these lock down: an earlier header regex matched the word ``user`` or
``assistant`` *inside prose*, so ``### assistant`` leaked into 79% of the
question fields and questions ended with fragments like ``"\\n\\nas"``.  The model
was then trained on questions no human would ever type.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from boxz.data.prepare_v3 import (dedupe, parse_any, question_key,
                                  split_by_question)

SYSTEM = ("You are Boxz, a helpful assistant for web games. "
          "Give correct answers to the user and to the assistant.")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_hashed_headers_with_system_turns(tmp_path):
    text = (
        "### system\n" + SYSTEM + "\n\n"
        "### user\nWhat is HTML?\n\n"
        "### assistant\nHTML is the markup language.\n\n---\n\n"
        "### system\n" + SYSTEM + "\n\n"
        "### user\nWhat is CSS?\n\n"
        "### assistant\nCSS styles the page.\n"
    )
    p = _write(tmp_path, "d.txt", text)
    ex = parse_any(p, "general")
    assert len(ex) == 2
    assert ex[0].question == "What is HTML?"
    assert ex[0].answer == "HTML is the markup language."
    assert ex[1].question == "What is CSS?"


def test_prose_containing_role_words_is_not_a_header(tmp_path):
    """'user' and 'assistant' inside a sentence must not split a block."""
    text = (
        "### user\n"
        "Every beginner user asks this.\n\n"
        "### assistant\n"
        "Tell the user to read the docs.\n"
    )
    ex = parse_any(_write(tmp_path, "d.txt", text), "general")
    assert len(ex) == 1
    assert ex[0].question == "Every beginner user asks this."


def test_no_structural_markers_survive(tmp_path):
    text = (
        "### user\n₹12,600 is divided equally among 28 classmates. What is the share?\n\n"
        "### assistant\n₹12,600 ÷ 28 = ₹450 each.\n"
    )
    ex = parse_any(_write(tmp_path, "d.txt", text), "math")
    assert "###" not in ex[0].question and "###" not in ex[0].answer
    assert not ex[0].question.endswith("as")


def test_bare_headers_work(tmp_path):
    text = "user\nWhat is 2 + 2?\n\nassistant\n2 + 2 = 4.\n\n---\n\n" \
           "user\nWhat is 3 + 3?\n\nassistant\n3 + 3 = 6.\n"
    ex = parse_any(_write(tmp_path, "m.txt", text), "math")
    assert len(ex) == 2
    assert ex[0].question == "What is 2 + 2?"


def test_code_fence_indentation_is_preserved(tmp_path):
    text = ("### user\nShow a loop.\n\n### assistant\nHere:\n\n"
            "```javascript\nfor (let i = 0; i < 3; i++) {\n    go(i);\n}\n```\n")
    ex = parse_any(_write(tmp_path, "c.txt", text), "coding")
    assert "    go(i);" in ex[0].answer, "indent inside a fence must survive"


def test_duplicates_removed_but_distinct_kept():
    from boxz.data.prepare_v3 import Example

    rows = [Example("What is 2 + 2?", "2 + 2 = 4.", "math"),
            Example("what is  2 + 2 ?", "2 + 2 = 4.", "math"),   # exact dup
            Example("What is 3 + 3?", "2 + 2 = 4.", "math")]     # same answer,
    kept, stats = dedupe(rows)                                   # different Q
    assert len(kept) == 2
    assert stats["exact_duplicate_pairs_removed"] == 1


def test_split_never_shares_a_question():
    from boxz.data.prepare_v3 import Example

    rows = [Example(f"What is {i} + {i}?", f"{2*i}", "math") for i in range(60)]
    rows += [Example(f"what is  {i} + {i} ?", f"{2*i}", "math") for i in range(60)]
    train, val = split_by_question(rows, 0.2)
    tr = {question_key(e.question) for e in train}
    va = {question_key(e.question) for e in val}
    assert not (tr & va)
    assert len(train) + len(val) <= len(rows)


@pytest.mark.parametrize("path", ["maths.txt", "coding.txt", "Data.txt"])
def test_real_files_have_no_structural_leak(path):
    p = Path(path)
    if not p.exists():
        pytest.skip(f"{path} not present")
    ex = parse_any(p, "x")
    assert ex, f"{path} parsed to nothing"
    leaked = [e for e in ex if "###" in e.question or "###" in e.answer]
    assert not leaked, f"{path}: {len(leaked)} rows still contain '###'"


def test_sft_encoding_never_double_terminates():
    """Regression: every row used to end '<|end|><|end|>'.

    render_pieces() already appends <|end|> to assistant turns, so appending a
    second one taught the model to double-terminate.  The template's <|end|>
    must also keep its up-weight.
    """
    from boxz.training.sft_v3 import _encode_example

    class _Tok:
        """Minimal stand-in: one id per character, plus special ids."""

        pad_id = 0
        _specials = {"<|end|>": 7, "<|user|>": 8, "<|assistant|>": 9}

        def special_id(self, name):
            return self._specials[name]

        def encode(self, text):
            return [ord(c) % 1000 + 10 for c in text]

    ex = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]}
    x, y, w = _encode_example(ex, _Tok(), max_len=512, end_weight=4.0)
    seq = [int(i) for i in x] + [int(y[-1])]
    ends = _Tok._specials["<|end|>"]
    pairs = sum(1 for i in range(len(seq) - 1) if seq[i] == ends and seq[i + 1] == ends)
    assert pairs == 0, "consecutive <|end|> tokens in target sequence"
    assert seq[-1] == ends, "sequence should still terminate"
    assert float(w[-1]) == 4.0, "end token should keep its up-weight"

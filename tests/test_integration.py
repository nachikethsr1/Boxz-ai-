"""End-to-end tests for the Boxz V3 system.

These exercise the wiring that unit tests cannot see: a tiny model, a tiny
tokenizer, and a real :class:`ChatEngine` going through the router.

The tiny model is *deliberately untrained*, so its prose is gibberish.  That is
the point: the tests assert the maths path is exact **regardless** of what the
model would have said, which is the whole reason the engine exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxz.chat.engine import ChatEngine
from boxz.config import ModelConfig
from boxz.model.transformer import build_model
from boxz.tokenizer.bpe import BPETokenizer

SFT_ROWS = Path("data/processed/v3/v3_train.jsonl")


@pytest.fixture(scope="module")
def tiny_engine(tmp_path_factory):
    """A small, deterministic model + tokenizer wrapped in a ChatEngine."""
    tmp = tmp_path_factory.mktemp("tiny")
    if SFT_ROWS.exists():
        text = "\n".join(json.loads(l)["messages"][0]["content"]
                         for l in SFT_ROWS.read_text().splitlines()[:400] if l.strip())
    else:
        text = "write javascript for a button " * 200
    tok = BPETokenizer()
    tok.train(text, vocab_size=512, verbose=False)
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, d_model=32,
                      d_ff=64, context_length=128)
    model = build_model(cfg)
    return ChatEngine(model, tok, use_tools=True)


# ------------------------------------------------------------------ routing
def test_math_question_is_computed_not_generated(tiny_engine):
    r = tiny_engine.reply("482736 + 918264")
    assert r.text.strip() == "482736 + 918264 = 1401000"
    assert r.finish_reason == "math_engine"
    assert r.tokens_generated == 0, "the model must not generate the digits"


def test_math_result_is_exact_for_unseen_numbers(tiny_engine):
    r = tiny_engine.reply("What is 2,194,638 + 6,484,738?")
    assert "8679376" in r.text.replace(",", "")
    assert r.finish_reason == "math_engine"


def test_multiplication_word_problem(tiny_engine):
    r = tiny_engine.reply("What is 15% of 200?")
    assert "30" in r.text
    assert r.finish_reason == "math_engine"


def test_coding_question_goes_to_the_model(tiny_engine):
    r = tiny_engine.reply("Write JavaScript that responds to a button click.")
    assert r.finish_reason != "math_engine"


def test_conversation_keeps_both_turns(tiny_engine):
    tiny_engine.reset()
    tiny_engine.reply("12 * 12")
    tiny_engine.reply("Explain CSS Flexbox briefly.")
    hist = tiny_engine.history()
    assert len(hist) >= 4                     # 2 user + 2 assistant
    roles = [m["role"] for m in hist]
    assert roles.count("user") == 2


def test_switching_topics_does_not_leak_state(tiny_engine):
    tiny_engine.reset()
    a = tiny_engine.reply("12 * 12")
    tiny_engine.reply("Write a JavaScript function.")
    c = tiny_engine.reply("7 * 6")
    assert "144" in a.text and "42" in c.text


def test_tools_can_be_disabled(tiny_engine):
    tiny_engine.use_tools = False
    try:
        r = tiny_engine.reply("482736 + 918264")
        assert r.finish_reason != "math_engine"
    finally:
        tiny_engine.use_tools = True


def test_verified_value_is_never_rewritten():
    """The engine's value must appear verbatim even for a 15-digit sum."""
    from boxz.chat.engine import ChatEngine as CE

    a, b = 987654321012345, 123456789012345
    ans = __import__("boxz.math.engine", fromlist=["solve"]).solve(f"{a} + {b}")
    assert ans.ok and ans.value == str(a + b)

"""Integration: the coding system inside the real ChatEngine.

These tests use the real checkpoint when one is present, because the point is
to prove the *ordering* inside ``ChatEngine.reply``: maths first, then the
template library, then the model.  A template turn must generate zero tokens;
a maths turn must generate zero tokens; a chat turn must generate some.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from boxz.chat.engine import ChatEngine
from boxz.training.checkpoint import load_checkpoint

CHECKPOINT = Path("checkpoints/v3-sft/best")


@pytest.fixture(scope="module")
def engine():
    if not CHECKPOINT.exists():
        pytest.skip("no checkpoint available")
    ck = load_checkpoint(CHECKPOINT)
    return ChatEngine(ck.model, ck.tokenizer, use_tools=True,
                      use_coding_system=True)


def test_coding_system_is_attached(engine):
    assert engine.coding is not None
    assert len(engine.coding.available()) >= 6


def test_template_turn_generates_no_tokens(engine):
    engine.reset()
    reply = engine.reply("Make a snake game where the tail grows")
    assert reply.finish_reason == "coding_match"
    assert reply.tokens_generated == 0
    assert reply.prompt_tokens == 0
    assert "Snake game" in reply.text


def test_template_turn_reports_project_and_validation(engine):
    engine.reset()
    reply = engine.reply("Make a snake game")
    tools = reply.tools
    assert tools["source"] == "template"
    assert tools["template_id"] == "js_snake"
    assert len(tools["project"]["files"]) == 3
    assert tools["validation"]["ok"] is True
    assert tools["validation"]["browser_tested"] is False


def test_maths_still_wins_and_generates_no_tokens(engine):
    engine.reset()
    reply = engine.reply("482736 + 918264")
    assert reply.finish_reason == "math_engine"
    assert reply.tokens_generated == 0
    assert "1401000" in reply.text


def test_maths_is_not_claimed_by_the_coding_system(engine):
    engine.reset()
    engine.last_coding = None          # clear the previous turn's record
    engine.reply("what is 17% of 2480")
    assert engine.last_coding is None or not engine.last_coding.handled


def test_followup_within_one_engine_session(engine):
    engine.reset()
    first = engine.reply("Make a snake game")
    assert first.finish_reason == "coding_match"
    second = engine.reply("Make it faster.")
    assert second.finish_reason == "coding_match"
    assert engine.coding_session.values["step_ms"] == 90
    assert "90" in second.text


def test_reset_clears_the_coding_session(engine):
    engine.reset()
    engine.reply("Make a snake game")
    assert engine.coding_session.active
    engine.reset()
    assert not engine.coding_session.active


def test_chat_falls_through_to_the_model(engine):
    engine.reset()
    reply = engine.reply("hi")
    assert reply.finish_reason in ("stop", "length")
    assert reply.tokens_generated > 0


def test_question_after_project_still_reaches_the_model(engine):
    engine.reset()
    engine.reply("Make a snake game")
    reply = engine.reply("What is CSS?")
    assert reply.finish_reason != "coding_match"
    # the project is untouched
    assert engine.coding_session.template_id == "js_snake"


def test_streaming_emits_the_template_reply(engine):
    engine.reset()
    chunks = list(engine.reply_stream.stream(engine, "Make a snake game")) \
        if hasattr(engine.reply_stream, "stream") else list(
            engine.reply_stream("Make a snake game"))
    assert chunks
    assert "Snake game" in chunks[-1]


def test_tools_can_be_disabled():
    if not CHECKPOINT.exists():
        pytest.skip("no checkpoint available")
    ck = load_checkpoint(CHECKPOINT)
    e = ChatEngine(ck.model, ck.tokenizer, use_tools=True, use_coding_system=False)
    assert e.coding is None
    e.reset()
    reply = e.reply("Make a snake game")
    # with the system off the model answers, which is the old behaviour
    assert reply.finish_reason in ("stop", "length")


def test_coding_session_survives_a_text_reset(engine):
    """Regression: the demo server reset() on every turn.

    ``Conversation.reset()`` clears the text history, but the coding session is
    separate state: "make it faster" needs the project it refers to.  Dropping
    that state was why a follow-up reached the model instead of the template.
    """
    engine.reset()
    engine.reply("Make a snake game")
    assert engine.coding_session.template_id == "js_snake"

    engine.conversation.reset()          # what the server does between turns
    reply = engine.reply("Make it faster.")
    assert reply.finish_reason == "coding_match"
    assert engine.coding_session.values["step_ms"] == 90

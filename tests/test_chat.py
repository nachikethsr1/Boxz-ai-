"""Chat layer: formatting, history management, truncation, engine behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from boxz.chat.engine import ChatEngine
from boxz.chat.format import Conversation, Message
from boxz.config import GenerationConfig, ModelConfig
from boxz.model.transformer import build_model
from boxz.tokenizer.bpe import BPETokenizer


def test_message_role_validation():
    with pytest.raises(ValueError):
        Message("robot", "hi")


def test_conversation_render_format():
    conv = Conversation(system="be helpful")
    conv.add_user("hi").add_assistant("hello")
    text = conv.render()
    assert text == "<|system|>be helpful\n<|user|>hi\n<|assistant|>hello<|end|>"
    assert conv.render_prompt().endswith("<|assistant|>")


def test_truncation_keeps_recent_turns():
    tok = BPETokenizer().train("hello world " * 100, vocab_size=300)
    conv = Conversation(system="sys")
    for i in range(10):
        conv.add_user(f"question number {i} " * 5)
        conv.add_assistant(f"answer {i} " * 5)
    conv.add_user("final question")
    trimmed = conv.truncated(tok, max_tokens=60)
    assert len(trimmed.messages) < len(conv.messages)
    assert trimmed.messages[-1].content == "final question"
    assert len(tok.encode(trimmed.render_prompt())) <= 60


def test_history_roundtrip():
    conv = Conversation.from_list([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ])
    assert conv.to_list() == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]


@pytest.fixture(scope="module")
def engine():
    tok = BPETokenizer().train("hello world the cat sat on the mat. " * 100, vocab_size=300)
    # A tiny context forces the conversation truncation policy to run, which is
    # exactly what the multi-turn test needs to observe.
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, d_model=32,
                      d_ff=64, context_length=160)
    m = build_model(cfg, seed=0)
    return ChatEngine(m, tok, system_prompt="be helpful", max_context_tokens=96)


def test_engine_reply_comes_from_model(engine):
    res = engine.reply("hello", GenerationConfig(max_new_tokens=8, temperature=0.7, seed=0))
    assert res.tokens_generated == 8
    assert isinstance(res.text, str)
    # the recorded history must match what the model produced
    assert engine.history()[-1] == {"role": "assistant", "content": res.text}


def test_engine_multi_turn_prompt_contains_history(engine):
    engine.reset()
    engine.reply("first", GenerationConfig(max_new_tokens=4, temperature=0.0))
    res2 = engine.reply("second", GenerationConfig(max_new_tokens=4, temperature=0.0))
    assert "<|user|>first" in res2.prompt
    assert "<|assistant|>" in res2.prompt
    assert "<|user|>second" in res2.prompt


def test_engine_stops_at_turn_marker(engine):
    res = engine.reply("hi", GenerationConfig(max_new_tokens=30, temperature=1.0, seed=1))
    assert "<|user|>" not in res.text
    assert "<|end|>" not in res.text
    assert "<|assistant|>" not in res.text


def test_engine_output_is_printable(engine):
    """Control characters must never appear in replies (sampling mask)."""
    for seed in range(5):
        res = engine.reply("tell me something", GenerationConfig(max_new_tokens=20, seed=seed))
        assert not any(ord(ch) < 32 and ch not in "\n\t" for ch in res.text), repr(res.text)


def test_reset_clears_history(engine):
    engine.reply("hi", GenerationConfig(max_new_tokens=2))
    assert engine.history()
    engine.reset()
    assert engine.history() == []

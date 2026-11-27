"""Corpus quality filtering, v2 instruction set, and context-budget safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from boxz.data.build_instruct_v2 import build_examples, split_examples
from boxz.data.code_quality import FilterConfig, clean_documents, quality_report


# ------------------------------------------------------- code quality filter
def _doc(code: str, path: str = "x.html") -> dict:
    return {"repo": "r/r", "path": path, "licence": "MIT", "commit": "abc",
            "chars": len(code), "code": code}


def test_noise_tags_are_removed():
    docs = [_doc('<head><meta property="og:title" content="x">'
                 '<title>page</title><link rel="stylesheet" href="a.css">'
                 + "const a = 1;\n" * 80)]
    out = clean_documents(docs, FilterConfig(min_chars=50), verbose=False)
    assert out
    assert "<meta" not in out[0]["code"]
    assert "<title" not in out[0]["code"]
    assert "const a = 1;" in out[0]["code"]


def test_boilerplate_lines_are_dropped():
    boiler = "renderer.setSize( window.innerWidth, window.innerHeight );"
    docs = [_doc(boiler + "\n" + f"const v{i} = {i};\n" * 60, f"f{i}.html") for i in range(60)]
    out = clean_documents(docs, FilterConfig(min_repeats=10, min_chars=50), verbose=False)
    assert out
    for d in out:
        assert boiler not in d["code"]


def test_near_duplicates_are_dropped():
    body = "\n".join(f"const value{i} = compute({i});" for i in range(80))
    docs = [_doc(body, f"f{i}.html") for i in range(10)]
    out = clean_documents(docs, FilterConfig(min_chars=50), verbose=False)
    assert len(out) == 1, "identical files must collapse to one"


def test_short_files_dropped():
    docs = [_doc("tiny", "a.html"), _doc("x = 1;\n" * 200, "b.html")]
    out = clean_documents(docs, FilterConfig(min_chars=500), verbose=False)
    assert len(out) == 1


def test_quality_report_shape():
    rep = quality_report("data/raw/code_corpus/corpus.json")
    assert rep["files_before"] > rep["files_after"]
    assert rep["chars_after"] < rep["chars_before"]
    assert 0 < rep["reduction"] < 1


# --------------------------------------------------------- v2 instruction set
def test_v2_builds_enough_examples():
    exs = build_examples()
    assert len(exs) >= 110
    assert all(len(e["messages"]) == 2 for e in exs)
    assert all(e["messages"][1]["content"].strip() for e in exs)


def test_v2_questions_unique_and_answers_not_pooled():
    """v1 mapped many questions onto a few answers; v2 must not."""
    exs = build_examples()
    qs = [e["messages"][0]["content"].lower() for e in exs]
    assert len(qs) == len(set(qs))
    answers = [e["messages"][1]["content"] for e in exs]
    # at most a handful of collisions are acceptable (short chit-chat replies)
    assert len(set(answers)) >= len(answers) * 0.85


def test_v2_split_is_disjoint_deterministic_and_covers_all():
    exs = build_examples()
    a = split_examples(exs)
    b = split_examples(exs)
    key = lambda part: [json_dump(x) for x in part]
    assert key(a) == key(b), "split must be deterministic"
    train, dev, test = a
    assert len(train) + len(dev) + len(test) == len(exs)
    sets = [{json_dump(x) for x in part} for part in a]
    assert not (sets[0] & sets[1]), "train/dev overlap"
    assert not (sets[0] & sets[2]), "train/test overlap"
    assert not (sets[1] & sets[2]), "dev/test overlap"


def json_dump(x: dict) -> str:
    import json
    return json.dumps(x, sort_keys=True)


# ------------------------------------------------------- context budget safety
@pytest.fixture(scope="module")
def tok():
    from boxz.tokenizer.bpe import BPETokenizer
    corpus = Path("data/raw/tinyshakespeare.txt")
    text = corpus.read_text(encoding="utf-8")[:200_000] if corpus.exists() else "abc def " * 5000
    return BPETokenizer().train(text, vocab_size=300)


def test_truncation_never_exceeds_context(tok):
    """Regression: a long single message produced a 407-token prompt for a
    128-token model, because truncation could only drop whole messages."""
    from boxz.chat.format import Conversation

    for ctx in (64, 128):
        conv = Conversation().add_user("x " * 600 + "final question")
        trimmed = conv.truncated(tok, ctx)
        n = len(tok.encode(trimmed.render_prompt()))
        assert n <= ctx, f"{n} tokens > context {ctx}"


def test_truncation_keeps_the_tail_of_a_long_message(tok):
    from boxz.chat.format import Conversation

    conv = Conversation().add_user("noise " * 400 + " how do i center a div?")
    trimmed = conv.truncated(tok, 64)
    assert "center" in trimmed.messages[-1].content.lower() or \
           "div" in trimmed.messages[-1].content.lower()


def test_truncation_drops_old_turns_first(tok):
    from boxz.chat.format import Conversation

    conv = Conversation()
    for i in range(8):
        conv.add_user(f"question number {i} " * 4)
        conv.add_assistant(f"answer number {i} " * 4)
    conv.add_user("final question")
    trimmed = conv.truncated(tok, 80)
    assert trimmed.messages[-1].content == "final question"
    assert len(trimmed.messages) < len(conv.messages)

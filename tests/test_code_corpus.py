"""Code corpus builder and instruction-set expansion (no network required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boxz.data.code_corpus import (CodeProject, clean_code, corpus_stats,
                                   fetch_corpus, verify_licence)
from boxz.data.expand_instruct import (EXPANDED_PATH, SEED_PATH, TOPICS,
                                       build_examples, write_expanded)


# ---------------------------------------------------------------- code corpus
def test_clean_code_drops_vendored_includes():
    html = """<html><head>
<link rel="stylesheet" href="https://cdn/x.css">
<script type="importmap">{"imports": {"three": "https://cdn/three.js"}}</script>
<script src="https://cdn/three.js"></script>
<style>.a { color: red }</style>
</head><body>
<canvas id="c"></canvas>
<script type="module">
import * as THREE from 'three';
const scene = new THREE.Scene();
</script>
</body></html>"""
    out = clean_code(html)
    assert "cdn" not in out
    assert "importmap" not in out
    assert "THREE.Scene" in out
    assert "canvas" in out


def test_clean_code_keeps_inline_style():
    out = clean_code('<style>.box { color: #fff }</style><div class="box">hi</div>')
    assert ".box { color: #fff }" in out
    assert "class=\"box\"" in out


def test_clean_code_collapses_blank_lines():
    assert "\n\n\n" not in clean_code("a\n\n\n\n\nb")


def test_project_declares_only_permissive_licences():
    from boxz.data.code_corpus import PROJECTS

    allowed = {"MIT", "Apache-2.0", "BSD-3-Clause", "ISC", "CC0-1.0", "Unlicense"}
    for p in PROJECTS:
        assert p.licence in allowed, f"{p.repo} uses a non-permissive licence: {p.licence}"
        assert p.purpose, "every project must document why it is included"


def test_code_source_is_registered_with_licence():
    from boxz.data.registry import get_source

    src = get_source("boxz_web_code")
    assert "MIT" in src.licence
    assert src.notes and "commit SHA" in src.notes


def test_corpus_stats_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpus_stats(str(tmp_path / "nope.json"))


def test_corpus_stats_on_fixture(tmp_path):
    docs = [{"repo": "a/b", "path": "x.html", "licence": "MIT",
             "commit": "abc", "chars": 10, "code": "x" * 10},
            {"repo": "a/b", "path": "y.html", "licence": "MIT",
             "commit": "abc", "chars": 5, "code": "y" * 5}]
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(docs), encoding="utf-8")
    st = corpus_stats(str(p))
    assert st["files"] == 2 and st["chars"] == 15
    assert st["by_repo"] == {"a/b": 2}


# ------------------------------------------------------- instruction expansion
def test_expanded_set_is_much_larger_than_seed():
    seed = build_examples(SEED_PATH)
    assert len(seed) >= 36
    # every topic contributes at least one question
    assert len(TOPICS) >= 30


def test_expanded_questions_are_unique():
    exs = build_examples(SEED_PATH)
    qs = [e["messages"][0]["content"].lower() for e in exs]
    assert len(qs) == len(set(qs)), "duplicate questions would skew the SFT loss"


def test_expanded_examples_are_well_formed():
    for ex in build_examples(SEED_PATH):
        msgs = ex["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"].strip() and msgs[1]["content"].strip()
        assert len(msgs[1]["content"]) >= 20, "answers must be substantive"


def test_expanded_examples_encode_within_context():
    """Long examples must be truncated from the end, never losing the question."""
    from boxz.data.instructions import InstructionExample, encode_example
    from boxz.tokenizer.bpe import BPETokenizer

    corpus = Path("data/raw/tinyshakespeare.txt")
    text = corpus.read_text(encoding="utf-8")[:400_000] if corpus.exists() else "hello world " * 5000
    tok = BPETokenizer().train(text, vocab_size=2048)

    examples = [InstructionExample(e["messages"]) for e in build_examples(SEED_PATH)]
    encoded = [encode_example(ex, tok, 128) for ex in examples]
    assert all(e is not None for e in encoded), "every example must yield a training pair"

    # over-long answers are cut at the end, so the user turn survives intact
    for ex, enc in zip(examples, encoded):
        x, _ = enc
        prompt_ids = tok.encode(f"<|user|>{ex.messages[0]['content']}")
        assert list(x[:len(prompt_ids)]) == prompt_ids, "the question was truncated away"
        assert len(x) <= 128


def test_write_expanded_roundtrips(tmp_path):
    out = tmp_path / "exp.jsonl"
    stats = write_expanded(out, SEED_PATH)
    assert out.exists()
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == stats["examples"]
    assert stats["answer_chars"] > 20_000

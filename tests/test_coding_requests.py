"""Tests for the ``coding_requests.txt`` parser.

The parser is the front door for every routing example, so a line it silently
ignores is a routing example that never fires.  These tests therefore assert
that malformed input *raises*.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from boxz.coding.requests import (
    DEFAULT_REQUEST_FILE,
    RequestBlock,
    parse_coding_requests,
)

REQUESTS = Path(DEFAULT_REQUEST_FILE)


# ---------------------------------------------------------------- real file
def test_real_file_parses():
    corpus = parse_coding_requests(REQUESTS)
    assert corpus.blocks, "the shipped request file produced no blocks"
    assert corpus.categories, "no categories parsed"


def test_real_file_has_expected_shape():
    corpus = parse_coding_requests(REQUESTS)
    summary = corpus.summary()
    assert summary["templates"] > 50, summary
    assert summary["requests"] > 500, summary
    assert summary["categories"] >= 8, summary


def test_every_block_has_requests():
    corpus = parse_coding_requests(REQUESTS)
    empty = [b.id for b in corpus.blocks if not b.requests]
    assert empty == [], f"blocks with no requests: {empty}"


def test_template_ids_are_unique():
    corpus = parse_coding_requests(REQUESTS)
    ids = [b.id for b in corpus.blocks]
    assert len(ids) == len(set(ids))


def test_no_duplicate_requests():
    corpus = parse_coding_requests(REQUESTS)
    seen = [r.lower() for b in corpus.blocks for r in b.requests]
    assert len(seen) == len(set(seen))


def test_ids_follow_the_documented_rule():
    """Lowercase letters, digits and underscore only."""
    import re
    corpus = parse_coding_requests(REQUESTS)
    ok = re.compile(r"^[a-z0-9_]+$")
    bad = [b.id for b in corpus.blocks if not ok.match(b.id)]
    assert bad == [], f"invalid ids: {bad}"


# ---------------------------------------------------------------- views
def test_block_kinds_and_intents():
    corpus = parse_coding_requests(REQUESTS)
    for b in corpus.templates:
        assert b.intent == "coding" and b.template_id == b.id and b.reason is None
    for b in corpus.clarifications:
        assert b.intent == "clarify" and b.template_id is None and b.reason == b.id
    for b in corpus.unsupported:
        assert b.intent == "unsupported" and b.template_id is None


def test_flat_records_cover_every_request():
    corpus = parse_coding_requests(REQUESTS)
    records = corpus.flat_records()
    assert len(records) == corpus.request_count()
    for r in records:
        assert set(r) == {"utterance", "intent", "template_id", "reason",
                          "category", "block_kind"}
        assert r["utterance"].strip()


def test_routing_edge_cases_exist():
    corpus = parse_coding_requests(REQUESTS)
    kinds = {b.id: b.kind for b in corpus.blocks}
    assert kinds.get("game_type_unknown") == "CLARIFY"
    assert kinds.get("language_not_supported") == "NO_TEMPLATE"
    assert kinds.get("framework_not_supported") == "NO_TEMPLATE"


# ---------------------------------------------------------------- errors
def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "req.txt"
    p.write_text(body, encoding="utf-8")
    return p


def test_request_before_any_block_is_an_error(tmp_path):
    p = _write(tmp_path, "=== CATEGORY: games\n- orphan request\n")
    with pytest.raises(ValueError, match="before any === TEMPLATE"):
        parse_coding_requests(p)


def test_block_before_category_is_an_error(tmp_path):
    p = _write(tmp_path, "=== TEMPLATE: foo\n- make foo\n")
    with pytest.raises(ValueError, match="before any === CATEGORY"):
        parse_coding_requests(p)


def test_duplicate_block_id_is_an_error(tmp_path):
    p = _write(tmp_path,
               "=== CATEGORY: c\n=== TEMPLATE: t\n- one\n=== TEMPLATE: t\n- two\n")
    with pytest.raises(ValueError, match="duplicate block id"):
        parse_coding_requests(p)


def test_duplicate_request_is_an_error(tmp_path):
    p = _write(tmp_path,
               "=== CATEGORY: c\n=== TEMPLATE: t\n- same\n- Same\n")
    with pytest.raises(ValueError, match="duplicate request"):
        parse_coding_requests(p)


def test_unknown_line_is_an_error(tmp_path):
    p = _write(tmp_path, "=== CATEGORY: c\n=== TEMPLATE: t\nnot a request\n")
    with pytest.raises(ValueError, match="unrecognised line"):
        parse_coding_requests(p)


def test_block_with_no_requests_is_an_error(tmp_path):
    p = _write(tmp_path, "=== CATEGORY: c\n=== TEMPLATE: empty\n")
    with pytest.raises(ValueError, match="no requests"):
        parse_coding_requests(p)


def test_comments_and_blank_lines_are_ignored(tmp_path):
    p = _write(tmp_path,
               "# a comment\n\n=== CATEGORY: c\n\n=== TEMPLATE: t\n- real\n\n")
    corpus = parse_coding_requests(p)
    assert corpus.request_count() == 1
    assert corpus.templates[0].requests == ["real"]


def test_category_carries_down_to_later_blocks(tmp_path):
    p = _write(tmp_path,
               "=== CATEGORY: first\n=== TEMPLATE: a\n- one\n"
               "=== TEMPLATE: b\n- two\n")
    corpus = parse_coding_requests(p)
    assert [b.category for b in corpus.blocks] == ["first", "first"]


def test_new_category_attaches_to_following_block(tmp_path):
    p = _write(tmp_path,
               "=== CATEGORY: first\n=== TEMPLATE: a\n- one\n"
               "=== CATEGORY: second\n=== TEMPLATE: b\n- two\n")
    corpus = parse_coding_requests(p)
    assert [b.category for b in corpus.blocks] == ["first", "second"]

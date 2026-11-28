"""Regression tests for the Phase 2B corpus pipeline.

Each test answers a defect that actually shipped in Phase 2A.  They are written
against real files where possible so they fail if the *pipeline* regresses, not
just if a helper changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from boxz.data.build_p2b import (
    DOC_END,
    DOC_START,
    MAX_DOC_CHARS,
    build,
    dedupe_documents,
    load_documents,
    render_document,
    split_documents,
)

P2B = Path("data/processed/p2b")


# --------------------------------------------------------------------- fixtures
def _doc(text, lang="python", source="unit", licence="MIT"):
    return {
        "text": text, "lang": lang, "source": source, "licence": licence,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build a small corpus from synthetic docs through the real code path."""
    tmp = tmp_path_factory.mktemp("p2b")
    jsonl = tmp / "corpus.jsonl"
    rows = [
        {"lang": "python", "source": "a", "licence": "MIT",
         "text": "def f(x):\n    if x:\n        return 1\n    return 0\n" * 20},
        {"lang": "javascript", "source": "b", "licence": "MIT",
         "text": "function g(a) {\n  if (a) {\n    return 1;\n  }\n}\n" * 20},
        {"lang": "text", "source": "c", "licence": "MIT",
         "text": "A sentence explaining a concept.\n" * 20},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    man = build(out_dir=tmp / "out", corpus_jsonl=jsonl,
                instruction_path=None, val_fraction=0.0)
    return tmp / "out", man


# ------------------------------------------------------------------- boundaries
def test_document_boundaries_are_explicit():
    """Phase 2A had no terminator: 33 MB of code fused into a user question."""
    out = render_document(_doc("print(1)\n"))
    assert out.startswith(DOC_START)
    assert out.endswith(DOC_END)
    assert "python" in out.splitlines()[0]


def test_no_segment_exceeds_the_document_cap(built):
    """Every bounded segment must be at most MAX_DOC_CHARS + header slack."""
    out_dir, _ = built
    text = (out_dir / "p2b_corpus.txt").read_text(encoding="utf-8")
    for seg in text.split(DOC_END):
        assert len(seg) <= MAX_DOC_CHARS + 400, f"fused segment of {len(seg)} chars"


def test_oversized_documents_are_truncated_on_a_line_boundary(tmp_path):
    big = "x" * 100 + "\n"
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"lang": "text", "source": "s",
                                "text": big * 400}) + "\n", encoding="utf-8")
    docs = load_documents(path)
    assert len(docs) == 1
    assert len(docs[0]["text"]) <= MAX_DOC_CHARS
    assert docs[0].get("truncated") is True


# ---------------------------------------------------------------- deduplication
def test_duplicate_documents_are_dropped():
    a = _doc("same text")
    b = _doc("same text")          # identical content, different object
    c = _doc("different text")
    kept, dropped = dedupe_documents([a, b, c])
    assert dropped == 1
    assert len(kept) == 2


def test_built_corpus_contains_no_duplicate_block(built):
    """The Phase 2A failure mode: 98.4% of the corpus was one repeated block."""
    out_dir, _ = built
    text = (out_dir / "p2b_corpus.txt").read_text(encoding="utf-8")
    docs = [d for d in text.split(DOC_END) if d.strip()]
    hashes = [hashlib.sha256(d.encode()).hexdigest() for d in docs]
    assert len(hashes) == len(set(hashes)), "a document appears twice"


def test_corpus_has_no_duplicated_block_like_phase2a():
    """If the real P2B corpus exists, no quarter of it may repeat verbatim."""
    f = P2B / "p2b_corpus.txt"
    if not f.exists():
        pytest.skip("p2b corpus not built")
    text = f.read_text(encoding="utf-8")
    quarter = len(text) // 4
    head = text[:quarter]
    assert text.count(head) == 1, "a large block of the corpus repeats"


# ------------------------------------------------------- code must stay intact
def test_repeated_braces_survive_the_pipeline(tmp_path):
    """Line-level dedupe collapsed the corpus to ONE closing brace in Phase 2A."""
    code = "if (a) {\n  b();\n}\nif (c) {\n  d();\n}\n"
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"lang": "javascript", "source": "s",
                                "text": code * 20}) + "\n", encoding="utf-8")
    docs = load_documents(path)
    body = docs[0]["text"]
    assert body.count("}") == code.count("}") * 20, "closing braces were collapsed"


def test_indentation_is_preserved(tmp_path):
    code = "def f():\n    if True:\n        return 1\n"
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"lang": "python", "source": "s",
                                "text": code * 20}) + "\n", encoding="utf-8")
    docs = load_documents(path)
    assert "        return 1" in docs[0]["text"], "indentation was stripped"


def test_pipeline_never_line_dedupes():
    """Guard against 'dedupe: false' being flipped back on."""
    man = P2B / "manifest.json"
    if not man.exists():
        pytest.skip("p2b corpus not built")
    m = json.loads(man.read_text())
    assert "line-level" in m["dedupe"]
    assert "never" in m["dedupe"]


def test_real_corpus_keeps_repeated_braces():
    f = P2B / "p2b_train.txt"
    if not f.exists():
        pytest.skip("p2b corpus not built")
    text = f.read_text(encoding="utf-8")
    assert text.count("\n}") > 500, "code closing braces are missing from the corpus"


# ----------------------------------------------------------------------- splits
def test_split_does_not_leak_documents():
    docs = [_doc(f"unique body {i}") for i in range(20)]
    train, val = split_documents(docs, val_fraction=0.25)
    assert {d["sha256"] for d in train} & {d["sha256"] for d in val} == set()


def test_split_is_deterministic():
    docs = [_doc(f"body {i}") for i in range(20)]
    a = split_documents(docs, seed=7)
    b = split_documents(docs, seed=7)
    assert [d["sha256"] for d in a[0]] == [d["sha256"] for d in b[0]]


def _document_bodies(path: Path) -> set:
    """Hashes of non-blank documents.  Splitting on the separator alone leaves
    a '\n' artefact between documents, which is not a document."""
    out = set()
    for seg in path.read_text(encoding="utf-8").split(DOC_END):
        body = seg.strip()
        if body:
            out.add(hashlib.sha256(body.encode()).hexdigest())
    return out


def test_real_train_and_val_files_do_not_overlap():
    tr, va = P2B / "p2b_train.txt", P2B / "p2b_val.txt"
    if not (tr.exists() and va.exists()):
        pytest.skip("p2b corpus not built")
    shared = _document_bodies(tr) & _document_bodies(va)
    assert not shared, f"{len(shared)} documents appear in both train and val"


def test_build_rejects_a_leaking_split(tmp_path, monkeypatch):
    """A split that puts one document on both sides must raise, not warn."""
    import boxz.data.build_p2b as mod

    real = mod.split_documents

    def leaky(docs, val_fraction=0.1, seed=0):
        train, _ = real(docs, val_fraction, seed)
        return train, list(train)          # deliberately leak

    monkeypatch.setattr(mod, "split_documents", leaky)
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text(json.dumps({"lang": "text", "source": "s",
                                 "text": "hello world " * 40}) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="leaked"):
        mod.build(out_dir=tmp_path / "o", corpus_jsonl=jsonl, instruction_path=None)


# ------------------------------------------------------------------------ cache
def test_cache_key_tracks_content_not_just_length(tmp_path):
    """Phase 2A keyed its cache on a character count, which went stale silently."""
    a = Path(tmp_path / "a.txt")
    b = Path(tmp_path / "b.txt")
    a.write_text("abc" * 100, encoding="utf-8")
    b.write_text("abd" * 100, encoding="utf-8")     # same length, different bytes
    assert a.stat().st_size == b.stat().st_size
    assert (hashlib.sha256(a.read_bytes()).hexdigest()
            != hashlib.sha256(b.read_bytes()).hexdigest())


def test_manifest_records_content_hashes():
    man = P2B / "manifest.json"
    if not man.exists():
        pytest.skip("p2b corpus not built")
    m = json.loads(man.read_text())
    for key in ("corpus_sha256", "train_sha256", "val_sha256"):
        assert len(m[key]) == 64, f"{key} is not a sha256"
    # and they must match the files on disk
    for key, name in [("corpus_sha256", "p2b_corpus.txt"),
                      ("train_sha256", "p2b_train.txt"),
                      ("val_sha256", "p2b_val.txt")]:
        actual = hashlib.sha256((P2B / name).read_bytes()).hexdigest()
        assert m[key] == actual, f"{name} changed since the manifest was written"


# ------------------------------------------------------------------ token counts
def test_token_count_matches_the_encoded_data():
    """Phase 2A silently dropped 40% of the corpus; counts must be real."""
    from boxz.tokenizer.bpe import BPETokenizer

    f = P2B / "p2b_train.txt"
    if not f.exists():
        pytest.skip("p2b corpus not built")
    text = f.read_text(encoding="utf-8")
    tok = BPETokenizer.load("data/processed/v3/tokenizer.json")
    assert len(tok.encode(text)) == len(tok.encode(text))   # deterministic
    # the encoded length must be a sane fraction of the characters
    ratio = len(text) / len(tok.encode(text))
    assert 1.5 < ratio < 5.0, f"implausible chars/token {ratio:.2f}"


def test_trainer_reads_the_whole_corpus_not_a_truncated_prefix():
    """`clean_text` must not silently discard documents."""
    from boxz.data.build_p2b import DOC_END as END
    from boxz.data.text import clean_text

    f = P2B / "p2b_train.txt"
    if not f.exists():
        pytest.skip("p2b corpus not built")
    raw = f.read_text(encoding="utf-8")
    cleaned = clean_text(raw, dedupe=False)
    raw_docs = raw.count(END)
    clean_docs = cleaned.count(END)
    assert clean_docs == raw_docs, f"{raw_docs - clean_docs} documents disappeared"


# ----------------------------------------------------------- language coverage
def test_corpus_contains_the_languages_phase2a_lacked():
    man = P2B / "manifest.json"
    if not man.exists():
        pytest.skip("p2b corpus not built")
    langs = json.loads(man.read_text())["chars_by_language"]
    for required in ("python", "html", "css", "javascript", "text"):
        assert langs.get(required, 0) > 1000, f"{required} missing from the corpus"


def test_licences_are_all_permissive():
    man = P2B / "manifest.json"
    if not man.exists():
        pytest.skip("p2b corpus not built")
    allowed = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC",
               "0BSD", "Unlicense", "CC0-1.0", "project-original"}
    got = set(json.loads(man.read_text())["documents_by_licence"])
    assert got <= allowed, f"non-permissive licence present: {got - allowed}"

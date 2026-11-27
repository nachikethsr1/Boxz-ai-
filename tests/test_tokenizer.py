"""Tokenizer tests: lossless round-trip, special tokens, persistence, BPE learning."""

from __future__ import annotations

import json

import pytest

from boxz.tokenizer.bpe import DEFAULT_SPECIALS, BPETokenizer, bytes_to_unicode, unicode_to_bytes

CORPUS = """
The quick brown fox jumps over the lazy dog. The quick brown fox is quick.
def add(a, b):
    return a + b

function updatePlayer(dt) {
  player.x += speed * dt;
}
"""


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer().train(CORPUS * 20, vocab_size=400, min_frequency=2)


def test_byte_maps_are_inverse():
    for b in range(256):
        assert unicode_to_bytes(bytes_to_unicode()[b]) == bytes([b])


def test_special_token_ids_are_stable():
    t = BPETokenizer()
    assert t.pad_id == 0 and t.unk_id == 1 and t.eos_id == 3
    assert t.special_to_id["<|assistant|>"] == 5
    assert len(DEFAULT_SPECIALS) == 8


def test_roundtrip_is_lossless(tok):
    samples = [
        "hello world",
        "",
        "  leading and trailing  ",
        "unicode: héllo — 日本語 🎮 ✓",
        "code:\n\tif (x) { y++; }\n",
        CORPUS,
    ]
    for s in samples:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, f"round trip failed for {s!r}"


def test_unknown_text_never_fails(tok):
    # Arbitrary bytes that are not valid UTF-8 cannot appear in a str, but any
    # str (including control characters and rare scripts) must encode.
    s = "".join(chr(c) for c in range(1, 1000) if chr(c).isprintable())
    assert tok.decode(tok.encode(s)) == s


def test_special_tokens_are_atomic(tok):
    text = "<|user|>hi<|end|>"
    ids = tok.encode(text)
    assert tok.special_id("<|user|>") in ids
    assert tok.special_id("<|end|>") in ids
    assert tok.decode(ids) == "hi"                      # specials hidden by default
    assert tok.decode(ids, skip_special_tokens=False) == text


def test_specials_disabled_treats_as_text(tok):
    ids = tok.encode("<|user|>", allowed_special=False)
    assert all(i >= tok.n_specials for i in ids)
    assert tok.decode(ids) == "<|user|>"


def test_vocab_layout(tok):
    assert tok.vocab_size == tok.n_specials + 256 + len(tok.merges)
    assert len(tok.id_to_token) == 256 + len(tok.merges)
    # every learned token is the concatenation of its merge parents
    for i, (a, b) in enumerate(tok.merges):
        assert tok.id_to_token[256 + i] == tok.id_to_token[a - tok.n_specials] + tok.id_to_token[b - tok.n_specials]


def test_bpe_compresses_better_than_bytes(tok):
    text = "The quick brown fox jumps over the lazy dog. " * 10
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_persistence_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.merges == tok.merges
    for s in ["The quick brown fox", "function updatePlayer"]:
        assert loaded.encode(s) == tok.encode(s)
    # the plain-text vocabulary view is also written
    assert path.with_suffix(".vocab.txt").exists()


def test_deterministic_training():
    a = BPETokenizer().train(CORPUS * 5, vocab_size=300)
    b = BPETokenizer().train(CORPUS * 5, vocab_size=300)
    assert a.merges == b.merges


def test_respects_vocab_size():
    for v in (300, 512, 1000):
        t = BPETokenizer().train(CORPUS * 30, vocab_size=v, min_frequency=1)
        assert t.vocab_size <= v
        assert t.vocab_size > 264


def test_too_small_vocab_rejected():
    with pytest.raises(ValueError):
        BPETokenizer().train(CORPUS, vocab_size=100)


def test_decode_offsets(tok):
    text, spans = tok.decode_with_offsets(tok.encode("hello world"))
    assert text == "hello world"
    assert len(spans) == len(tok.encode("hello world"))
    assert spans[0][0] == 0 and spans[-1][1] == len(text.encode())


def test_stats(tok):
    stats = tok.stats()
    assert stats["vocab_size"] == tok.vocab_size
    assert stats["merges"] == len(tok.merges)
    assert 1.0 <= stats["mean_token_bytes"] <= stats["max_token_bytes"]

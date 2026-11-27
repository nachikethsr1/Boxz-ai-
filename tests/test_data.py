"""Data pipeline tests: cleaning, splitting, batching, dataset registry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxz.config import DataConfig
from boxz.data.corpus import TokenizedCorpus, build_corpus, prepare_dataset, render_messages
from boxz.data.registry import get_source, list_sources
from boxz.data.text import clean_text
from boxz.tokenizer.bpe import BPETokenizer


@pytest.fixture(scope="module")
def corpus_and_tok(tmp_path_factory):
    text = ("Boxz is a small language model. " * 400) + ("<|user|>hello<|end|>" * 50)
    cfg = DataConfig(vocab_size=300, val_fraction=0.1)
    tmp = tmp_path_factory.mktemp("data")
    cfg.cache_dir = str(tmp)
    tok = BPETokenizer().train(text, vocab_size=300)
    corpus = build_corpus(text, tok, val_fraction=0.1)
    return corpus, tok


def test_clean_text_normalises():
    dirty = "line one\r\nline two   with\tspaces\x00\r\n\r\n\r\n\r\nline three"
    out = clean_text(dirty, dedupe=False)
    assert "\r" not in out and "\x00" not in out
    assert "\n\n\n" not in out
    assert "line two with spaces" in out


def test_clean_text_dedupes_lines():
    out = clean_text("a\nb\na\nb\nc", dedupe=True)
    assert out == "a\nb\nc"


def test_clean_text_truncates_and_checks_min():
    assert len(clean_text("x" * 100, max_chars=10)) == 10
    with pytest.raises(ValueError):
        clean_text("short", min_chars=100)


def test_split_sizes(corpus_and_tok):
    corpus, _ = corpus_and_tok
    total = corpus.n_tokens("train") + corpus.n_tokens("val")
    assert total == len(corpus.tokens)
    val_frac = corpus.n_tokens("val") / total
    assert 0.08 < val_frac < 0.12


def test_batches_are_shifted_by_one(corpus_and_tok):
    corpus, _ = corpus_and_tok
    x, y = corpus.batch(4, 32, "train", np.random.default_rng(0))
    assert x.shape == (4, 32) and y.shape == (4, 32)
    # y is the next-token target: for each row, data[s+1:s+33] == y
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])


def test_batches_do_not_exceed_vocab(corpus_and_tok):
    corpus, tok = corpus_and_tok
    x, y = corpus.batch(8, 16, "train", np.random.default_rng(1))
    assert x.max() < tok.vocab_size and y.max() < tok.vocab_size


def test_sequential_batches_are_deterministic(corpus_and_tok):
    corpus, _ = corpus_and_tok
    a = [x for x, _ in corpus.sequential_batches(2, 16, "val")]
    b = [x for x, _ in corpus.sequential_batches(2, 16, "val")]
    assert len(a) == len(b)
    for x1, x2 in zip(a, b):
        np.testing.assert_array_equal(x1, x2)


def test_batch_raises_when_split_too_short(tmp_path):
    tok = BPETokenizer().train("abcdef " * 50, vocab_size=300)
    corpus = build_corpus("abcdef " * 3, tok, val_fraction=0.5)
    with pytest.raises(ValueError):
        corpus.batch(2, 128, "val")


def test_seeded_batches_reproducible(corpus_and_tok):
    corpus, _ = corpus_and_tok
    x1, y1 = corpus.batch(4, 16, "train", np.random.default_rng(7))
    x2, y2 = corpus.batch(4, 16, "train", np.random.default_rng(7))
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_render_messages_format():
    text = render_messages([
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert text == "<|system|>be nice<|user|>hi<|assistant|>hello<|end|>"


def test_corpus_renderer_matches_chat_template():
    """There must be only one conversation format in the codebase."""
    from boxz.chat.template import render_messages as chat_render
    from boxz.data.corpus import render_messages as corpus_render

    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert corpus_render(msgs) == chat_render(msgs)


def test_pretraining_corpus_excludes_sft_holdout():
    """Held-out instruction rows must never be trained on (leakage guard).

    Retraining and then scoring the model on examples it already saw would make
    the reported SFT holdout loss a memorisation score rather than a
    generalisation measurement.
    """
    from boxz.data.corpus import _resolve_text
    from boxz.training.sft import holdout_indices
    import json

    src = Path("data/sources/boxz_instruct.jsonl")
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = holdout_indices(len(rows), 8, 3)

    text, *_ = _resolve_text(DataConfig(dataset=str(src)))
    for i in idx:
        answer = rows[int(i)]["messages"][-1]["content"]
        assert answer not in text, f"holdout row {i} leaked into the corpus"

    # positive control: with the guard disabled the rows are present
    text_all, *_ = _resolve_text(DataConfig(dataset=str(src), exclude_sft_holdout=False))
    assert all(rows[int(i)]["messages"][-1]["content"] in text_all for i in idx)


def test_registry_declares_licences():
    sources = list_sources()
    assert sources, "at least one dataset must be declared"
    for s in sources:
        assert s.licence, f"{s.name} must declare a licence"
        assert s.title and s.notes


def test_tinyshakespeare_is_registered():
    src = get_source("tinyshakespeare")
    assert "public domain" in src.licence.lower()


def test_prepare_dataset_end_to_end(tmp_path):
    src = tmp_path / "mini.txt"
    src.write_text("Hello boxz. " * 500, encoding="utf-8")
    cfg = DataConfig(dataset=str(src), vocab_size=300, cache_dir=str(tmp_path / "cache"))
    corpus, tok = prepare_dataset(cfg, verbose=False)
    assert corpus.n_tokens("train") > 100
    assert tok.vocab_size <= 300
    # cached tokenizer is reused on the second call
    _, tok2 = prepare_dataset(cfg, verbose=False)
    assert tok2.merges == tok.merges


def test_multi_dataset_resolution(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("alpha beta gamma " * 100)
    b.write_text("delta epsilon " * 100)
    from boxz.data.corpus import _resolve_text

    cfg = DataConfig(dataset=[str(a), str(b)])
    text, name, files, fmt = _resolve_text(cfg)
    assert fmt == "mixed" and "alpha" in text and "delta" in text
    assert len(files) == 2


def test_corpus_save(corpus_and_tok, tmp_path):
    corpus, _ = corpus_and_tok
    d = corpus.save(tmp_path / "out")
    assert (d / "tokens_train.npy").exists()
    stats = json.loads((d / "corpus_stats.json").read_text())
    assert stats["tokens_train"] > 0

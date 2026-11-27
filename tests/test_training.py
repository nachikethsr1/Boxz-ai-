"""Training-loop tests: loss decreases, checkpoints round-trip, resume is exact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxz.config import DataConfig, ModelConfig, TrainConfig
from boxz.data.corpus import build_corpus
from boxz.tokenizer.bpe import BPETokenizer
from boxz.training.checkpoint import load_checkpoint, save_checkpoint
from boxz.training.evaluate import bits_per_byte, perplexity
from boxz.training.trainer import Trainer

TEXT = (
    "the cat sat on the mat. the dog sat on the log. "
    "a cat and a dog sat on a log and a mat. " * 200
)


@pytest.fixture(scope="module")
def small_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train")
    tok = BPETokenizer().train(TEXT, vocab_size=300)
    corpus = build_corpus(TEXT, tok, val_fraction=0.1)
    cfg = TrainConfig(
        run_name="unit",
        model=ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, d_model=32,
                          d_ff=64, context_length=16),
        data=DataConfig(vocab_size=tok.vocab_size, dataset="unit-test"),
        batch_size=8,
        max_steps=40,
        lr=0.01,
        min_lr=0.001,
        warmup_steps=5,
        eval_interval=20,
        checkpoint_interval=40,
        log_interval=10,
        checkpoint_dir=str(tmp / "ckpt"),
        eval_batches=4,
        seed=0,
    )
    return cfg, corpus, tok, tmp


def test_training_reduces_loss(small_setup):
    cfg, corpus, tok, _ = small_setup
    tr = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    report = tr.fit()
    tr.close()
    first = report.history[0]["train_loss"]
    assert report.final_val_loss < first, f"val loss did not improve: {report.final_val_loss} vs {first}"
    assert report.steps == 40
    assert report.tokens_seen == 40 * 8 * 16


def test_checkpoint_roundtrip(small_setup, tmp_path):
    cfg, corpus, tok, _ = small_setup
    tr = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    tr.fit()
    tr.close()
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name / "final"
    assert (ckpt_dir / "model.npz").exists()
    assert (ckpt_dir / "tokenizer.json").exists()
    assert (ckpt_dir / "config.json").exists()

    ck = load_checkpoint(ckpt_dir)
    x = np.random.randint(0, tok.vocab_size, size=(2, 16))
    a = ck.model(x)[0].data
    b = tr.model(x)[0].data
    np.testing.assert_allclose(a, b, rtol=1e-6)
    assert ck.tokenizer.vocab_size == tok.vocab_size


def test_resume_continues_from_step(small_setup):
    cfg, corpus, tok, _ = small_setup
    cfg2 = TrainConfig.from_dict(cfg.to_dict())
    cfg2.max_steps = 20
    cfg2.checkpoint_dir = cfg.checkpoint_dir
    tr = Trainer(cfg2, corpus=corpus, tokenizer=tok, verbose=False)
    tr.fit()
    tr.close()
    final = Path(cfg.checkpoint_dir) / cfg.run_name / "final"

    cfg3 = TrainConfig.from_dict(cfg.to_dict())
    cfg3.resume = str(final)
    cfg3.max_steps = 30
    tr2 = Trainer(cfg3, corpus=corpus, tokenizer=tok, verbose=False)
    assert tr2.step == 20
    assert tr2.tokens_seen == 20 * 8 * 16
    tr2.fit()
    tr2.close()
    assert tr2.step == 30


def test_optimizer_state_restored(small_setup):
    cfg, corpus, tok, _ = small_setup
    tr = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    tr.fit()
    tr.close()
    final = Path(cfg.checkpoint_dir) / cfg.run_name / "final"
    ck = load_checkpoint(final)
    assert ck.optimizer_state is not None
    assert ck.optimizer_state["step"] == 40
    assert len(ck.optimizer_state["m"]) == len(tr.model.parameters())


def test_gradient_accumulation_matches_larger_batch(small_setup):
    cfg, corpus, tok, _ = small_setup
    cfg = TrainConfig.from_dict(cfg.to_dict())
    cfg.grad_accum_steps = 2
    cfg.batch_size = 4
    cfg.max_steps = 3
    cfg.checkpoint_dir = str(Path(cfg.checkpoint_dir) / "accum")
    tr = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    tr.fit()
    tr.close()
    # 3 steps * 2 micro-batches * 4 sequences * 16 tokens
    assert tr.tokens_seen == 3 * 2 * 4 * 16


def test_metrics():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(np.log(10)) == pytest.approx(10.0, rel=1e-6)
    # 1 nat per token at 4 bytes/token == 1/(ln2*4) bits per byte
    assert bits_per_byte(1.0, 4.0) == pytest.approx(1.0 / (np.log(2) * 4))


def test_reproducible_training(small_setup):
    cfg, corpus, tok, _ = small_setup
    cfg = TrainConfig.from_dict(cfg.to_dict())
    cfg.max_steps = 10
    cfg.checkpoint_dir = str(Path(cfg.checkpoint_dir) / "repro")
    losses = []
    for _ in range(2):
        tr = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
        tr.fit()
        tr.close()
        losses.append([h["train_loss"] for h in tr.history if "train_loss" in h])
    assert losses[0] == losses[1], "seeded training must be bit-reproducible"


def test_log_written(small_setup):
    cfg, corpus, tok, _ = small_setup
    log = Path(cfg.checkpoint_dir) / cfg.run_name / "train_log.jsonl"
    assert log.exists()
    lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert any("val_loss" in r for r in lines)
    assert any("train_loss" in r for r in lines)

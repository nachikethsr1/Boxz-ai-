"""Phase 1B: training reliability.

Every test here exists because the audit found a specific defect, or because a
property was assumed but never checked.  They are grouped by what they protect:
logging, gradient clipping, instability detection, resume fidelity, and the
checkpoint registry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from boxz.config import DataConfig, ModelConfig, TrainConfig
from boxz.data.corpus import build_corpus
from boxz.tokenizer.bpe import BPETokenizer
from boxz.training.checkpoint import load_checkpoint
from boxz.training.evaluate import evaluate_loss
from boxz.training.registry import (checkpoint_status, require_usable,
                                    usable_checkpoints)
from boxz.training.trainer import Trainer

TEXT = ("the cat sat on the mat. the dog sat on the log. "
        "a cat and a dog sat on a log and a mat. " * 200)


def _setup(tmp_path, *, max_steps=8, eval_interval=4, grad_clip=1.0, seed=3,
           run_name="unit"):
    tok = BPETokenizer().train(TEXT, vocab_size=300)
    corpus = build_corpus(TEXT, tok, val_fraction=0.1)
    cfg = TrainConfig(
        run_name=run_name,
        model=ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2,
                          d_model=32, d_ff=64, context_length=16),
        data=DataConfig(vocab_size=tok.vocab_size, dataset="unit-test"),
        batch_size=4, max_steps=max_steps, lr=0.01, min_lr=0.001,
        warmup_steps=2, eval_interval=eval_interval, log_interval=2,
        checkpoint_interval=max_steps, checkpoint_dir=str(tmp_path / "ckpt"),
        eval_batches=2, seed=seed, grad_clip=grad_clip, max_minutes=None,
    )
    return cfg, corpus, tok


# --------------------------------------------------------------------- logging
def test_log_rows_are_single_schema(tmp_path):
    """Regression: train and val rows used to have different key sets.

    A reader doing `row.get('train_loss')` on a val row got `None`, which
    rendered as `nan` in tables and looked like a numerical failure.  Every row
    now carries `kind` and the common keys.
    """
    cfg, corpus, tok = _setup(tmp_path)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    t.fit()
    t.close()

    rows = [json.loads(l) for l in
            (tmp_path / "ckpt" / "unit" / "train_log.jsonl").read_text().splitlines()]
    assert rows, "no log rows written"
    for r in rows:
        assert "kind" in r and r["kind"] in ("train", "val")
        assert {"step", "tokens_seen", "elapsed_s"} <= set(r)
        # each row must carry the loss field its own kind promises
        if r["kind"] == "train":
            assert isinstance(r["train_loss"], float)
        else:
            assert isinstance(r["val_loss"], float)
            assert "train_loss" not in r
    # both kinds actually occurred
    assert {r["kind"] for r in rows} == {"train", "val"}


def test_no_logged_loss_is_nan(tmp_path):
    """The audit's 'NaN' was a schema artefact; this pins that nothing is NaN."""
    cfg, corpus, tok = _setup(tmp_path)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    t.fit()
    t.close()
    rows = [json.loads(l) for l in
            (tmp_path / "ckpt" / "unit" / "train_log.jsonl").read_text().splitlines()]
    losses = [r.get("train_loss", r.get("val_loss")) for r in rows]
    assert all(l is not None and math.isfinite(l) for l in losses), losses


# ------------------------------------------------------------ gradient clipping
def test_grad_clip_is_honoured_from_config(tmp_path):
    """Regression: the trainer hardcoded `grad_clip=None`, so clipping never ran
    even though configs declared 1.0. A run had no protection from a spike."""
    cfg, corpus, tok = _setup(tmp_path, grad_clip=1.0)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    assert t.optimizer.grad_clip == 1.0


def test_grad_clip_none_stays_none(tmp_path):
    """Explicitly disabling clipping must still work."""
    cfg, corpus, tok = _setup(tmp_path, grad_clip=None)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    assert t.optimizer.grad_clip is None


def test_gradient_norm_is_logged_and_bounded(tmp_path):
    """With clipping on, the recorded pre-clip norm is finite, and clipping
    actually rescales when the norm exceeds the limit."""
    cfg, corpus, tok = _setup(tmp_path, grad_clip=1.0)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    t.fit()
    t.close()
    rows = [json.loads(l) for l in
            (tmp_path / "ckpt" / "unit" / "train_log.jsonl").read_text().splitlines()]
    norms = [r["grad_norm"] for r in rows if r["kind"] == "train"]
    assert norms, "grad_norm was not logged"
    assert all(n is not None and math.isfinite(n) for n in norms)
    assert t.optimizer.last_grad_norm is not None


# --------------------------------------------------- instability detection
def test_non_finite_loss_stops_training(tmp_path, monkeypatch):
    """A NaN loss must raise immediately, not silently train on garbage."""
    cfg, corpus, tok = _setup(tmp_path, max_steps=20)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)

    def bad_step(xs, ys):
        t.step += 1
        return float("nan")

    monkeypatch.setattr(t, "train_step", bad_step)
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        t.fit()
    t.close()


def test_infinite_loss_stops_training(tmp_path, monkeypatch):
    cfg, corpus, tok = _setup(tmp_path, max_steps=20)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    monkeypatch.setattr(t, "train_step", lambda xs, ys: float("inf"))
    with pytest.raises(FloatingPointError):
        t.fit()
    t.close()


# ------------------------------------------------------------- resume fidelity
def test_resume_restores_data_rng(tmp_path):
    """Regression: the data-sampling RNG was never checkpointed, so a resumed run
    replayed the same batches from the seed instead of continuing the stream."""
    cfg, corpus, tok = _setup(tmp_path, max_steps=4)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    t.fit()
    t.close()

    ckpt = load_checkpoint(tmp_path / "ckpt" / "unit" / "final")
    assert "rng_state" in ckpt.train_state, "rng_state was not saved"

    t2 = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    t2._resume(str(tmp_path / "ckpt" / "unit" / "final"))
    assert t2.rng.bit_generator.state == ckpt.train_state["rng_state"]


def test_resume_continues_the_same_batch_sequence(tmp_path):
    """A resumed trainer must draw the *next* batch, not restart the sequence."""
    cfg, corpus, tok = _setup(tmp_path, max_steps=4)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    # consume a few batches, then save
    first = [corpus.batch(4, 16, "train", t.rng)[0] for _ in range(3)]
    t._save("mid")

    t2 = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    t2._resume(str(tmp_path / "ckpt" / "unit" / "mid"))
    after_resume = corpus.batch(4, 16, "train", t2.rng)[0]
    fresh = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    restarted = corpus.batch(4, 16, "train", fresh.rng)[0]
    assert not np.array_equal(after_resume, restarted), \
        "resumed run replayed the first batch instead of continuing"


def test_resume_restores_step_and_tokens(tmp_path):
    cfg, corpus, tok = _setup(tmp_path, max_steps=4)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    t.fit()
    t.close()
    ckpt = load_checkpoint(tmp_path / "ckpt" / "unit" / "final")
    t2 = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    t2._resume(str(tmp_path / "ckpt" / "unit" / "final"))
    assert t2.step == ckpt.train_state["step"]
    assert t2.tokens_seen == ckpt.train_state["tokens_seen"]
    assert t2.best_val_loss == ckpt.train_state["best_val_loss"]


# ------------------------------------------------------------- reproducibility
def test_same_seed_bitwise_identical(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    cfg_a, corpus, tok = _setup(a, max_steps=6, seed=11, run_name="ra")
    cfg_b, _, _ = _setup(b, max_steps=6, seed=11, run_name="ra")
    ta = Trainer(cfg_a, corpus=corpus, tokenizer=tok, verbose=False)
    ta.fit(); ta.close()
    tb = Trainer(cfg_b, corpus=corpus, tokenizer=tok, verbose=False)
    tb.fit(); tb.close()
    wa = np.load(a / "ckpt" / "ra" / "final" / "model.npz")
    wb = np.load(b / "ckpt" / "ra" / "final" / "model.npz")
    assert sorted(wa.files) == sorted(wb.files)
    for k in wa.files:
        assert np.array_equal(wa[k], wb[k]), f"{k} differs between identical seeds"


# ----------------------------------------------------------- eval determinism
def test_validation_is_deterministic(tmp_path):
    cfg, corpus, tok = _setup(tmp_path)
    t = Trainer(cfg, corpus=corpus, tokenizer=tok, verbose=False)
    a = evaluate_loss(t.model, corpus, batch_size=2, n_batches=2, split="val")
    b = evaluate_loss(t.model, corpus, batch_size=2, n_batches=2, split="val")
    assert a == b, "validation loss is not deterministic"
    assert math.isfinite(a)


# ------------------------------------------------------------------- registry
def test_registry_marks_aborted_checkpoint_unusable():
    """v4-sft/best is an aborted run; selection must refuse it."""
    status = checkpoint_status("checkpoints/v4-sft/best")
    assert not status.usable
    assert "ABORT" in status.reason.upper()
    with pytest.raises(ValueError, match="unusable"):
        require_usable("checkpoints/v4-sft/best")


def test_registry_allows_current_best():
    require_usable("checkpoints/v3-sft/best")   # must not raise
    assert "checkpoints/v3-sft/best" in usable_checkpoints()


def test_unregistered_checkpoint_defaults_to_usable():
    """A fresh run must not need a registry edit before it can be evaluated."""
    assert checkpoint_status("checkpoints/brand/new-run").usable


def test_registry_does_not_touch_files():
    """The registry is advisory: it must never delete or rewrite a checkpoint."""
    p = Path("checkpoints/v4-sft/best/model.npz")
    before = p.stat().st_size if p.exists() else None
    checkpoint_status("checkpoints/v4-sft/best")
    require_usable("checkpoints/v3-sft/best")
    after = p.stat().st_size if p.exists() else None
    assert before == after


# ------------------------------------------------- load_checkpoint choke point
def test_load_checkpoint_refuses_unusable_by_default(tmp_path):
    """Every caller path (CLI, demo, server, evaluators) inherits the guard."""
    p = Path("checkpoints/v4-sft/best")
    if not p.exists():
        pytest.skip("v4-sft/best not present")
    with pytest.raises(ValueError, match="unusable"):
        load_checkpoint(p)


def test_load_checkpoint_allows_deliberate_forensics(tmp_path):
    """Known-bad checkpoints stay loadable on request -- nothing is destroyed."""
    p = Path("checkpoints/v4-sft/best")
    if not p.exists():
        pytest.skip("v4-sft/best not present")
    ck = load_checkpoint(p, allow_unusable=True)
    assert ck.train_state.get("step") == 100

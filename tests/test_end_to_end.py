"""End-to-end integration: data -> tokenizer -> train -> checkpoint -> generate.

These tests are the acceptance criteria of Phase 1-3.  They train a real (very
small) model from raw text and assert that the artefact behaves like a language
model rather than a lookup table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxz.config import DataConfig, GenerationConfig, ModelConfig, TrainConfig
from boxz.data.corpus import prepare_dataset
from boxz.inference.generator import generate_text
from boxz.training.checkpoint import load_checkpoint
from boxz.training.evaluate import evaluate
from boxz.training.trainer import Trainer

CORPUS = (
    "Boxz learns to predict the next token from real text. "
    "Training reduces the loss because the model adjusts its weights. "
    "A language model assigns probabilities to sequences of tokens. "
) * 300


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    corpus_file = tmp / "corpus.txt"
    corpus_file.write_text(CORPUS, encoding="utf-8")
    cfg = TrainConfig(
        run_name="e2e",
        model=ModelConfig(vocab_size=400, n_layer=2, n_head=2, d_model=32, d_ff=64, context_length=24),
        data=DataConfig(dataset=str(corpus_file), vocab_size=400, cache_dir=str(tmp / "cache")),
        batch_size=8, max_steps=60, lr=0.01, min_lr=0.001, warmup_steps=5,
        eval_interval=30, checkpoint_interval=60, log_interval=20,
        checkpoint_dir=str(tmp / "ckpt"), eval_batches=4, seed=7,
    )
    trainer = Trainer(cfg, verbose=False)
    report = trainer.fit()
    trainer.close()
    return cfg, report, Path(cfg.checkpoint_dir) / cfg.run_name / "final"


@pytest.mark.slow
def test_full_pipeline(trained):
    cfg, report, ckpt_dir = trained
    assert report.final_val_loss < 4.0, f"model did not learn: {report.final_val_loss}"
    assert (ckpt_dir / "model.npz").exists()
    assert (ckpt_dir / "tokenizer.json").exists()

    ck = load_checkpoint(ckpt_dir)
    assert ck.model.num_parameters() > 0
    assert ck.model_config.vocab_size == ck.tokenizer.vocab_size

    # generation produces a string decoded by the real tokenizer
    res = generate_text(ck.model, ck.tokenizer, "Boxz learns to",
                        GenerationConfig(max_new_tokens=12, temperature=0.7, seed=0))
    assert isinstance(res.text, str) and res.tokens_generated == 12


@pytest.mark.slow
def test_evaluation_metrics_are_sane(trained):
    cfg, report, ckpt_dir = trained
    ck = load_checkpoint(ckpt_dir)
    corpus, _ = prepare_dataset(ck.train_config.data, ck.tokenizer, verbose=False)
    metrics = evaluate(ck.model, corpus, batch_size=4, n_batches=4)
    assert metrics["val_loss"] > 0
    assert metrics["val_perplexity"] > 1
    assert 0 <= metrics["val_next_token_accuracy"] <= 1
    # trained model must beat a uniform predictor
    import math

    assert metrics["val_loss"] < math.log(ck.model_config.vocab_size)


@pytest.mark.slow
def test_generation_depends_on_prompt(trained):
    """A real LM produces different continuations for different prompts."""
    cfg, report, ckpt_dir = trained
    ck = load_checkpoint(ckpt_dir)
    gen = GenerationConfig(max_new_tokens=10, temperature=0.0)
    a = generate_text(ck.model, ck.tokenizer, "Boxz learns", gen)
    b = generate_text(ck.model, ck.tokenizer, "A language model", gen)
    assert a.ids != b.ids
    # ... and identical prompt -> identical output (determinism)
    c = generate_text(ck.model, ck.tokenizer, "Boxz learns", gen)
    assert a.ids == c.ids


def test_cli_help_runs():
    from boxz.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0

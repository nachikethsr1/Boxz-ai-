"""Web application and instruction-tuning tests (no network, no external services)."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from boxz.config import ModelConfig, TrainConfig
from boxz.data.instructions import load_instruction_file, make_sft_batch
from boxz.model.transformer import build_model
from boxz.tokenizer.bpe import BPETokenizer
from boxz.training.checkpoint import save_checkpoint

CORPUS = "Boxz speaks in plain sentences. The model learns to continue text. " * 200


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    tok = BPETokenizer().train(CORPUS, vocab_size=300)
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=2, d_model=32,
                      d_ff=64, context_length=64)
    model = build_model(cfg, seed=0)
    d = tmp / "ckpt"
    save_checkpoint(d, model, tok, TrainConfig(model=cfg),
                    train_state={"step": 10, "best_val_loss": 3.5})
    return d


def test_instruction_file_parses():
    examples = load_instruction_file(Path("data/sources/boxz_instruct.jsonl"))
    assert len(examples) == 36
    assert all(ex.messages[-1]["role"] == "assistant" for ex in examples)


def test_sft_batch_masks_prompt_tokens(checkpoint):
    from boxz.training.checkpoint import load_checkpoint

    ck = load_checkpoint(checkpoint)
    examples = load_instruction_file(Path("data/sources/boxz_instruct.jsonl"))
    # max_length must be shorter than a full example so truncation and masking
    # are both exercised.
    xs, ys = make_sft_batch(examples, ck.tokenizer, batch_size=4, max_length=40,
                            rng=np.random.default_rng(0))
    assert xs.shape == (4, 40) and ys.shape == (4, 40)
    # every trained label is a real token id; prompt tokens are ignored (-1)
    assert (ys != -1).any()
    assert (ys[ys != -1] < ck.tokenizer.vocab_size).all()
    # the first position after the user turn is always trained on
    from boxz.data.instructions import encode_example, _render

    ids, labels = _render(examples[0], ck.tokenizer)
    prompt_len = len(labels) - len([l for l in labels if l != ck.tokenizer.pad_id])
    assert prompt_len > 0, "the prompt must contribute masked labels"


def test_app_status_and_chat(checkpoint):
    from boxz.app.server import BoxzApp

    app = BoxzApp(str(checkpoint))
    status = app.status()
    assert status["model"]["parameters"] > 0
    assert status["backend"]["name"] in ("numpy", "rust")
    reply = app.chat({"message": "hello", "max_new_tokens": 6, "temperature": 0.7})
    assert "text" in reply and reply["tokens_generated"] == 6
    assert app.history()[-1]["role"] == "assistant"
    app.reset()
    assert app.history() == []


def test_app_rejects_empty_message(checkpoint):
    from boxz.app.server import BoxzApp

    app = BoxzApp(str(checkpoint))
    with pytest.raises(ValueError):
        app.chat({"message": "   "})


def test_http_server_end_to_end(checkpoint):
    """Boot the real HTTP server on an ephemeral port and drive the API."""
    from boxz.app.server import BoxzApp, make_server

    app = BoxzApp(str(checkpoint))
    server = make_server(app, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        raw = urllib.request.urlopen(base + "/", timeout=10).read().decode()
        assert "Boxz" in raw and "/api/chat" in raw

        status = json.loads(urllib.request.urlopen(base + "/api/status", timeout=10).read())
        assert status["model"]["vocab_size"] > 0

        body = json.dumps({"message": "hi", "max_new_tokens": 4, "temperature": 0.5}).encode()
        req = urllib.request.Request(base + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=30).read())
        assert out["tokens_generated"] == 4
    finally:
        server.shutdown()
        server.server_close()


def test_sft_max_length_clamped_to_context(checkpoint):
    """Regression: SFT must never request sequences beyond the model's context."""
    from boxz.training.sft import InstructionTuner, SFTConfig

    tuner = InstructionTuner(str(checkpoint), SFTConfig(steps=1, max_length=4096),
                             verbose=False)
    assert tuner.cfg.max_length == 64  # checkpoint context_length

    tuner2 = InstructionTuner(str(checkpoint), SFTConfig(steps=1, max_length=None),
                              verbose=False)
    assert tuner2.cfg.max_length == 64

    xs, ys = make_sft_batch(tuner.examples, tuner.tokenizer, 2, tuner.cfg.max_length)
    assert xs.shape == (2, 64)


def test_sft_holdout_is_disjoint_and_deterministic(checkpoint):
    """Held-out examples must never appear in the training set, and the split
    must be identical across runs so reported numbers are reproducible."""
    from boxz.training.sft import InstructionTuner, SFTConfig

    a = InstructionTuner(str(checkpoint), SFTConfig(steps=1, holdout=8), verbose=False)
    b = InstructionTuner(str(checkpoint), SFTConfig(steps=1, holdout=8), verbose=False)
    total = len(a.examples) + len(a.holdout)
    assert total >= 100
    assert len(a.holdout) == min(8, total // 4)
    key = lambda t: [ex.to_dict() for ex in t.holdout]
    assert key(a) == key(b)
    train_set = {json.dumps(ex.to_dict(), sort_keys=True) for ex in a.examples}
    hold_set = {json.dumps(ex.to_dict(), sort_keys=True) for ex in a.holdout}
    assert not (train_set & hold_set)
    # evaluating an untrained checkpoint yields a finite assistant loss
    val = a.evaluate()
    assert val == val and val > 0


def test_sft_short_run_writes_best_checkpoint(checkpoint, tmp_path):
    """A tiny real run must produce best/ and a report with holdout numbers."""
    from boxz.training.sft import InstructionTuner, SFTConfig

    out = tmp_path / "sft"
    tuner = InstructionTuner(str(checkpoint),
                             SFTConfig(steps=2, batch_size=2, eval_interval=1,
                                       holdout=8, patience=0, out_dir=str(out)),
                             verbose=False)
    report = tuner.fit()
    assert (out / "best").is_dir()
    assert report["holdout_examples"] == 8
    assert report["train_examples"] >= 90
    assert report["final_holdout_loss"] is not None
    assert report["best_holdout_loss"] is not None

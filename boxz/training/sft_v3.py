"""Instruction-tune a V3 pretrained checkpoint on the prepared V3 splits.

The existing :class:`~boxz.training.sft.InstructionTuner` re-splits whatever
file it is given and mixes in replay batches from the checkpoint's *pretraining*
config.  For V3 we already have a leak-free train/val split produced by
:mod:`boxz.data.prepare_v3`, and the pretraining corpus is the npm code dump --
replaying from it would flood SFT with third-party library code the model does
not need to imitate.

So this module is a thin, explicit trainer over ``v3_train.jsonl`` /
``v3_val.jsonl``:

* only assistant tokens contribute to the loss (prompt tokens are masked),
* the ``<|end|>`` token is weighted up so the model reliably terminates,
* validation loss is measured on rows *never* optimised,
* the best-by-val checkpoint is kept and the run stops on plateau.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..chat.template import render_pieces
from ..engine import no_grad
from ..engine.optim import AdamW, LRSchedule
from ..engine.rng import make_rng, seed_everything
from ..tokenizer.bpe import BPETokenizer
from ..training.checkpoint import load_checkpoint, save_checkpoint

__all__ = ["SFTV3Config", "SFTV3Result", "tune_v3"]


@dataclass
class SFTV3Config:
    train: str = "data/processed/v3/v3_train.jsonl"
    val: str = "data/processed/v3/v3_val.jsonl"
    steps: int = 1200
    batch_size: int = 8
    lr: float = 2e-4
    min_lr: float = 2e-5
    warmup_steps: int = 60
    weight_decay: float = 0.01
    eval_interval: int = 100
    patience: int = 5
    end_weight: float = 4.0
    max_length: int = 512
    seed: int = 3
    out_dir: str = "checkpoints/v3-sft"


@dataclass
class SFTV3Result:
    out_dir: str
    steps: int
    final_val_loss: float
    best_val_loss: float
    best_step: int
    train_seconds: float
    history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["history"] = self.history[-60:]
        return d


def _encode_example(ex: dict, tok: BPETokenizer, max_len: int,
                    end_weight: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (x, y, w) for one example, or None if it cannot fit.

    ``w`` is the per-token loss weight: 0 on prompt tokens, ``end_weight`` on
    ``<|end|>``, 1 elsewhere in the answer.
    """
    ids: List[int] = []
    weights: List[float] = []
    end_id = tok.special_id("<|end|>")
    for text, trainable in render_pieces(ex["messages"]):
        enc = tok.encode(text)
        ids.extend(enc)
        weights.extend([1.0 if trainable else 0.0] * len(enc))
        if trainable:
            # render_pieces appends no <|end|>; the template does it in
            # render_messages, so add it explicitly for a clean stop token.
            ids.append(end_id)
            weights.append(end_weight)
    if len(ids) < 2:
        return None
    if len(ids) > max_len + 1:
        return None
    x = np.asarray(ids[:-1], dtype=np.int64)
    y = np.asarray(ids[1:], dtype=np.int64)
    w = np.asarray(weights[1:], dtype=np.float32)
    return x, y, w


class _Batcher:
    def __init__(self, rows: Sequence[dict], tok: BPETokenizer, max_len: int,
                 end_weight: float, rng: np.random.Generator) -> None:
        self.tok, self.max_len, self.end_weight, self.rng = tok, max_len, end_weight, rng
        self.encoded = []
        self.dropped = 0
        for r in rows:
            e = _encode_example(r, tok, max_len, end_weight)
            if e is None:
                self.dropped += 1
            else:
                self.encoded.append(e)

    def batch(self, size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not self.encoded:
            return None
        idx = self.rng.integers(0, len(self.encoded), size=size)
        seqs = [self.encoded[int(i)] for i in idx]
        L = max(len(s[0]) for s in seqs)
        pad = self.tok.pad_id
        x = np.full((len(seqs), L), pad, dtype=np.int64)
        y = np.full((len(seqs), L), -1, dtype=np.int64)     # -1 = ignore
        w = np.zeros((len(seqs), L), dtype=np.float32)
        for i, (sx, sy, sw) in enumerate(seqs):
            n = len(sx)
            x[i, :n] = sx
            y[i, :n] = sy
            w[i, :n] = sw
        return x, y, w

    def all_batches(self, size: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Deterministic, non-overlapping batches for evaluation."""
        out = []
        for start in range(0, len(self.encoded), size):
            seqs = self.encoded[start:start + size]
            if not seqs:
                continue
            L = max(len(s[0]) for s in seqs)
            x = np.full((len(seqs), L), self.tok.pad_id, dtype=np.int64)
            y = np.full((len(seqs), L), -1, dtype=np.int64)
            w = np.zeros((len(seqs), L), dtype=np.float32)
            for i, (sx, sy, sw) in enumerate(seqs):
                n = len(sx)
                x[i, :n] = sx
                y[i, :n] = sy
                w[i, :n] = sw
            out.append((x, y, w))
        return out


def tune_v3(checkpoint: str, cfg: Optional[SFTV3Config] = None,
            verbose: bool = True) -> SFTV3Result:
    """Run instruction tuning and return measured results."""
    cfg = cfg or SFTV3Config()
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    seed_everything(cfg.seed)
    rng = make_rng(cfg.seed)

    ck = load_checkpoint(checkpoint)
    model = ck.model
    tok: BPETokenizer = ck.tokenizer
    ctx = int(ck.model_config.context_length)
    max_len = min(cfg.max_length, ctx)

    train_rows = [json.loads(l) for l in Path(cfg.train).read_text(encoding="utf-8").splitlines() if l.strip()]
    val_rows = [json.loads(l) for l in Path(cfg.val).read_text(encoding="utf-8").splitlines() if l.strip()]
    train_b = _Batcher(train_rows, tok, max_len, cfg.end_weight, rng)
    val_b = _Batcher(val_rows, tok, max_len, cfg.end_weight, rng)
    say(f"[sft-v3] train {len(train_b.encoded)} rows ({train_b.dropped} too long), "
        f"val {len(val_b.encoded)} rows ({val_b.dropped} too long), max_len {max_len}")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = LRSchedule(mode="cosine", lr=cfg.lr, min_lr=cfg.min_lr,
                       warmup_steps=cfg.warmup_steps, max_steps=cfg.steps)

    val_batches = val_b.all_batches(cfg.batch_size)

    def evaluate() -> float:
        model.eval()
        total_w = 0.0
        total = 0.0
        with no_grad():
            for x, y, w in val_batches:
                _, loss = model(x, y, loss_weights=w)
                n = float(w.sum())
                if n > 0:
                    total += float(loss.data) * n
                    total_w += n
                loss.free_graph()
        model.train()
        return total / max(total_w, 1e-9)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_step = 0
    bad = 0
    history: List[dict] = []
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        b = train_b.batch(cfg.batch_size)
        if b is None:
            raise RuntimeError("no training examples fit the context window")
        x, y, w = b
        opt.lr = sched.get(step - 1)
        model.zero_grad()
        _, loss = model(x, y, loss_weights=w)
        loss.backward()
        opt.step()
        model.zero_grad()
        train_loss = float(loss.data)
        loss.free_graph()

        if step % 25 == 0:
            say(f"  step {step:5d}/{cfg.steps}  train {train_loss:.4f}  lr {opt.lr:.2e}")

        if step % cfg.eval_interval == 0:
            v = evaluate()
            history.append({"step": step, "train_loss": round(train_loss, 4),
                            "val_loss": round(v, 4)})
            say(f"  eval step {step}: val_loss {v:.4f}")
            if v < best - 1e-4:
                best, best_step, bad = v, step, 0
                save_checkpoint(out / "best", model, tok, ck.train_config or ck.model_config,
                                train_state={"step": step, "best_val_loss": best,
                                             "stage": "sft_v3"})
                say(f"  new best val {v:.4f} -> {out / 'best'}")
            else:
                bad += 1
                if bad >= cfg.patience:
                    say(f"  early stop: no improvement for {bad} evals")
                    break

    final_val = history[-1]["val_loss"] if history else float("nan")
    save_checkpoint(out / "final", model, tok, ck.train_config or ck.model_config,
                    train_state={"step": step, "best_val_loss": best, "stage": "sft_v3"})
    return SFTV3Result(out_dir=str(out / "best"), steps=step,
                       final_val_loss=final_val, best_val_loss=best,
                       best_step=best_step, train_seconds=time.time() - t0,
                       history=history)

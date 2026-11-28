"""The Boxz training loop.

Responsibilities
----------------
* deterministic initialisation (seed -> weights -> data order -> dropout)
* learning-rate scheduling, gradient clipping, AdamW step
* periodic evaluation on a held-out split
* checkpointing + exact resume (model, optimizer moments, step, tokenizer)
* structured JSONL logs so every run is auditable after the fact

The loop is intentionally plain Python: a training step is
``forward -> backward -> (accumulate) -> clip -> AdamW -> zero_grad`` and
nothing is hidden behind a framework.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import TrainConfig
from ..data.corpus import TokenizedCorpus, prepare_dataset
from ..engine import no_grad
from ..engine.layers import Module
from ..engine.optim import AdamW, LRSchedule
from ..engine.rng import make_rng, seed_everything
from ..model.transformer import BoxzLM, build_model
from ..tokenizer.bpe import BPETokenizer
from .checkpoint import load_checkpoint, save_checkpoint
from .evaluate import evaluate_loss

__all__ = ["Trainer", "TrainReport"]


@dataclass
class TrainReport:
    run_dir: Path
    steps: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    tokens_seen: int
    seconds: float
    history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "steps": self.steps,
            "final_train_loss": round(self.final_train_loss, 4),
            "final_val_loss": round(self.final_val_loss, 4),
            "best_val_loss": round(self.best_val_loss, 4),
            "tokens_seen": self.tokens_seen,
            "seconds": round(self.seconds, 1),
            "tokens_per_second": round(self.tokens_seen / max(self.seconds, 1e-9), 1),
        }


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: Optional[BoxzLM] = None,
        corpus: Optional[TokenizedCorpus] = None,
        tokenizer: Optional[BPETokenizer] = None,
        verbose: bool = True,
    ) -> None:
        self.cfg = cfg
        self.verbose = verbose
        seed_everything(cfg.seed)
        self.rng = make_rng(cfg.seed)

        # -- data ---------------------------------------------------------
        if corpus is None or tokenizer is None:
            corpus, tokenizer = prepare_dataset(cfg.data, tokenizer, verbose=verbose)
            # The token array is only needed while batches are being sampled;
            # dropping the raw text now keeps ~300 MB free for activations,
            # which is what lets a larger micro-batch fit in 3.9 GB.
            import gc as _gc
            _gc.collect()
        self.corpus = corpus
        self.tokenizer = tokenizer
        if cfg.model.vocab_size != tokenizer.vocab_size:
            if verbose:
                print(f"[train] adjusting model vocab_size {cfg.model.vocab_size} -> {tokenizer.vocab_size}")
            cfg.model.vocab_size = tokenizer.vocab_size

        # -- model / optimiser -------------------------------------------
        self.model = model or build_model(cfg.model, seed=cfg.seed)
        if model is not None and model.cfg.vocab_size != tokenizer.vocab_size:
            raise ValueError("model and tokenizer vocabularies differ")
        if cfg.init_from and model is None:
            # Warm start: weights only.  Step, optimizer moments and the LR
            # schedule all begin fresh -- this is a new run that happens to
            # start from trained weights, not a continuation of the old one.
            ck = load_checkpoint(cfg.init_from)
            self.model.load_state_dict(ck.model.state_dict())
            if verbose:
                print(f"[train] warm start from {cfg.init_from} "
                      f"(weights only; step 0, fresh optimizer and schedule)")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
            # `grad_clip` was hardcoded to None here while the config declared
            # 1.0, so clipping silently never ran: the run had no protection
            # against a gradient spike.  Honour the configured value.
            grad_clip=cfg.grad_clip,
        )
        self.schedule = LRSchedule(mode=cfg.schedule, lr=cfg.lr, min_lr=cfg.min_lr,
                                   warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps)
        self.step = 0
        self.tokens_seen = 0
        self.history: List[Dict[str, float]] = []
        self.best_val_loss = float("inf")

        # -- output -------------------------------------------------------
        self.run_dir = Path(cfg.checkpoint_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train_log.jsonl"
        self._log_fh = None

        if cfg.resume:
            self._resume(cfg.resume)

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _log(self, record: Dict[str, Any]) -> None:
        self.history.append(record)
        if self._log_fh is None:
            self._log_fh = self.log_path.open("a", encoding="utf-8")
        self._log_fh.write(json.dumps(record) + "\n")
        self._log_fh.flush()

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    # resume
    # ------------------------------------------------------------------
    def _resume(self, path: str) -> None:
        ckpt = load_checkpoint(path)
        self.model.load_state_dict(ckpt.model.state_dict())
        if ckpt.optimizer_state:
            try:
                self.optimizer.load_state_dict(ckpt.optimizer_state)
            except Exception as exc:  # pragma: no cover - defensive
                self._say(f"[train] could not restore optimizer state: {exc}")
        self.step = int(ckpt.train_state.get("step", 0))
        self.tokens_seen = int(ckpt.train_state.get("tokens_seen", 0))
        self.best_val_loss = float(ckpt.train_state.get("best_val_loss", float("inf")))
        # Restore the data-sampling stream.  Without this, a resumed run
        # replays the same batches from the start of the seed sequence and a
        # "resumed" run is not the continuation of the original one.
        rng_state = ckpt.train_state.get("rng_state")
        if rng_state:
            try:
                self.rng.bit_generator.state = rng_state
            except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover
                self._say(f"[train] could not restore rng state: {exc}")
        self._say(f"[train] resumed from {path} at step {self.step}")

    # ------------------------------------------------------------------
    # one optimisation step
    # ------------------------------------------------------------------
    def train_step(self, xs: np.ndarray, ys: np.ndarray) -> float:
        lr = self.schedule.get(self.step)
        self.optimizer.lr = lr
        self.model.zero_grad()
        _, loss = self.model(xs, ys)
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
        # Release the graph eagerly.  free_graph() clears saved activations and
        # detaches the node chain, so the ~200 intermediate tensors of a step
        # are collected immediately by reference counting instead of surviving
        # until the next cyclic GC pass.
        loss.free_graph()
        self.step += 1
        self.tokens_seen += int(xs.size)
        return float(loss.data)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def fit(self) -> TrainReport:
        cfg = self.cfg
        t0 = time.time()
        self._say(self.model.summary())
        self._say(f"[train] run={cfg.run_name} steps={cfg.max_steps} batch={cfg.batch_size} "
                  f"ctx={cfg.model.context_length} lr={cfg.lr} schedule={cfg.schedule}")
        self._say(f"[train] tokens/step={cfg.batch_size * cfg.model.context_length:,} -> "
                  f"total planned {cfg.batch_size * cfg.model.context_length * cfg.max_steps:,}")

        running: List[float] = []
        last_val = float("inf")
        while self.step < cfg.max_steps:
            if cfg.max_minutes is not None and (time.time() - t0) / 60.0 > cfg.max_minutes:
                self._say(f"[train] wall-clock limit reached at step {self.step}")
                break
            if cfg.grad_accum_steps > 1:
                # _accumulated_step samples its own micro-batches; sampling one
                # here as well would silently discard `batch_size * ctx` tokens
                # on every single step.
                loss = self._accumulated_step(None, None)
            else:
                xs, ys = self.corpus.batch(cfg.batch_size, cfg.model.context_length,
                                           "train", self.rng)
                loss = self.train_step(xs, ys)
            if not math.isfinite(loss):
                # Without this the loop would keep stepping on NaN weights for
                # hours and only the final artefact would look wrong.  Stop at
                # the first non-finite loss and leave the last good checkpoint.
                raise FloatingPointError(
                    f"non-finite training loss {loss!r} at step {self.step + 1}; "
                    f"last good checkpoint in {self.run_dir}"
                )
            running.append(loss)
            if len(running) > 50:
                running.pop(0)

            if self.step % cfg.log_interval == 0 or self.step == 1:
                lr = self.schedule.get(self.step - 1)
                train_avg = float(np.mean(running))
                self._log({
                    "kind": "train",
                    "step": self.step,
                    "train_loss": round(train_avg, 4),
                    "lr": float(f"{lr:.3e}"),
                    "grad_norm": (round(self.optimizer.last_grad_norm, 4)
                                  if self.optimizer.last_grad_norm is not None else None),
                    "tokens_seen": self.tokens_seen,
                    "elapsed_s": round(time.time() - t0, 1),
                })
                self._say(f"step {self.step:5d}/{cfg.max_steps}  loss {train_avg:.4f}  "
                          f"lr {lr:.2e}  tokens {self.tokens_seen:>9,}  "
                          f"({round(time.time() - t0, 1):.0f}s)")

            if self.step % cfg.eval_interval == 0 and self.step > 0:
                val_loss = evaluate_loss(self.model, self.corpus, batch_size=max(4, cfg.batch_size // 2),
                                         n_batches=cfg.eval_batches, split="val")
                last_val = val_loss
                ppl = math.exp(min(val_loss, 50))
                self._log({
                    "kind": "val",
                    "step": self.step,
                    "val_loss": round(val_loss, 4),
                    "val_ppl": round(ppl, 2),
                    "tokens_seen": self.tokens_seen,
                    "elapsed_s": round(time.time() - t0, 1),
                })
                self._say(f"  eval step {self.step}: val_loss {val_loss:.4f}  ppl {ppl:.2f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best")
                    self._say(f"  new best val_loss {val_loss:.4f} -> {self.run_dir / 'best'}")

            if self.step % cfg.checkpoint_interval == 0 and self.step > 0:
                self._save(f"step_{self.step}")

        # final artefacts
        self._save("final")
        val_loss = (last_val if math.isfinite(last_val)
                    else evaluate_loss(self.model, self.corpus, cfg.batch_size, cfg.eval_batches, "val"))
        return TrainReport(
            run_dir=self.run_dir, steps=self.step,
            final_train_loss=float(np.mean(running)) if running else float("nan"),
            final_val_loss=val_loss, best_val_loss=self.best_val_loss,
            tokens_seen=self.tokens_seen, seconds=time.time() - t0, history=self.history,
        )

    def _accumulated_step(self, xs=None, ys=None) -> float:
        """Gradient accumulation over ``grad_accum_steps`` micro-batches."""
        n = self.cfg.grad_accum_steps
        self.optimizer.lr = self.schedule.get(self.step)
        self.model.zero_grad()
        losses = []
        for _ in range(n):
            xb, yb = self.corpus.batch(self.cfg.batch_size, self.cfg.model.context_length, "train", self.rng)
            _, loss = self.model(xb, yb)
            scaled = loss * (1.0 / n)
            scaled.backward()
            losses.append(float(loss.data))
            self.tokens_seen += int(xb.size)
            # Free this micro-batch's activations immediately.  Holding all N
            # graphs until after optimizer.step() multiplies peak activation
            # memory by N -- at ctx 512 that is the difference between fitting
            # in 3.9 GB and being OOM-killed.
            loss.free_graph()
            del scaled
        self.optimizer.step()
        self.model.zero_grad()
        self.step += 1
        return float(np.mean(losses))

    # ------------------------------------------------------------------
    def _save(self, tag: str) -> Path:
        d = self.run_dir / tag
        save_checkpoint(
            d, self.model, self.tokenizer, self.cfg,
            train_state={
                "step": self.step, "tokens_seen": self.tokens_seen,
                "best_val_loss": self.best_val_loss,
                "history": self.history[-200:],
                "rng_state": self.rng.bit_generator.state,
            },
            optimizer_state=self.optimizer.state_dict(),
        )
        return d

    def close(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

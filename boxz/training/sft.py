"""Instruction fine-tuning (Phase 4) on top of a pretrained checkpoint.

This is the same autograd engine and optimiser as pretraining; the differences
are (a) examples are conversations, (b) loss is masked to assistant tokens,
(c) learning rate and step count are much smaller because the model only needs to
adopt the conversation format and the assistant voice.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import ModelConfig, TrainConfig
from ..data.instructions import load_instruction_file, make_sft_batch
from ..engine.optim import AdamW, LRSchedule
from ..engine.rng import make_rng, seed_everything
from ..model.transformer import BoxzLM, build_model
from ..tokenizer.bpe import BPETokenizer
from .checkpoint import load_checkpoint, save_checkpoint

__all__ = ["InstructionTuner", "SFTConfig", "holdout_indices"]


def holdout_indices(n_examples: int, holdout: int = 16, seed: int = 3) -> np.ndarray:
    """Deterministic index set reserved for held-out evaluation.

    Used by both the SFT trainer and the benchmark suite so the number they
    report is measured on exactly the same examples, and none of those examples
    is ever optimised on.
    """
    if holdout <= 0 or n_examples < 4:
        return np.empty(0, dtype=np.int64)
    n_hold = min(int(holdout), n_examples // 4)
    order = np.random.default_rng(seed).permutation(n_examples)
    return np.sort(order[:n_hold]).astype(np.int64)


class SFTConfig:
    """Hyper-parameters for instruction tuning (kept separate from TrainConfig)."""

    def __init__(
        self,
        dataset: str = "data/sources/boxz_instruct_v2.jsonl",
        steps: int = 300,
        batch_size: int = 16,
        lr: float = 1e-4,
        min_lr: float = 1e-5,
        warmup_steps: int = 20,
        max_length: Optional[int] = None,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        eval_interval: int = 25,
        holdout: int = 16,
        patience: int = 8,
        replay_ratio: float = 0.15,
        replay_batches: int = 8,
        end_weight: float = 1.0,
        seed: int = 3,
        out_dir: str = "checkpoints/sft",
    ) -> None:
        self.dataset = dataset
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.eval_interval = eval_interval
        self.holdout = holdout
        self.patience = patience
        self.replay_ratio = replay_ratio
        self.replay_batches = replay_batches
        self.end_weight = end_weight
        self.seed = seed
        self.out_dir = out_dir

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class InstructionTuner:
    def __init__(self, checkpoint: str, cfg: Optional[SFTConfig] = None, verbose: bool = True) -> None:
        self.cfg = cfg or SFTConfig()
        self.verbose = verbose
        seed_everything(self.cfg.seed)
        self.rng = make_rng(self.cfg.seed)

        ck = load_checkpoint(checkpoint)
        self.model: BoxzLM = ck.model
        self.tokenizer: BPETokenizer = ck.tokenizer
        self.base_config = ck.model_config
        self.base_ckpt = checkpoint

        # The model cannot attend beyond its learned positional table, so the
        # example length must never exceed the checkpoint's context window.
        ctx = int(self.base_config.context_length)
        requested = self.cfg.max_length
        self.cfg.max_length = ctx if requested is None else min(int(requested), ctx)
        if requested is not None and int(requested) > ctx:
            self._say(f"[sft] max_length {requested} > context_length {ctx}; clamped to {ctx}")
        self._say(f"[sft] effective max_length={self.cfg.max_length} (context {ctx})")

        path = Path(self.cfg.dataset)
        if not path.exists():
            raise FileNotFoundError(f"instruction dataset not found: {path}")
        examples = load_instruction_file(path)
        if len(examples) < 4:
            raise ValueError("need at least four instruction examples")

        # A held-out split is what makes the SFT number meaningful: without it
        # the model simply memorises 36 answers (loss -> 0) and every reported
        # metric is contaminated.  The split is deterministic (seeded shuffle)
        # so results are reproducible.
        # Use the SAME deterministic train/dev/test split the evaluator uses, so
        # a reported held-out number cannot be computed on a row the optimiser
        # has seen.  ``holdout`` is kept as a legacy override.
        from ..data.build_instruct_v2 import split_examples

        use_shared_split = "boxz_instruct_v2" in path.name
        if not use_shared_split and self.cfg.holdout and self.cfg.holdout > 0:
            # legacy behaviour for small custom datasets: single holdout slice
            hold_idx = set(int(i) for i in
                           holdout_indices(len(examples), self.cfg.holdout, self.cfg.seed))
            self.examples = [ex for i, ex in enumerate(examples) if i not in hold_idx]
            self.dev = [examples[i] for i in range(len(examples)) if i in hold_idx]
            self.test: List = []
        else:
            self.examples, self.dev, self.test = split_examples(examples)
        self.holdout = self.dev
        self._say(f"[sft] split: {len(self.examples)} train / {len(self.dev)} dev / "
                  f"{len(self.test)} test")

        # Replay: mix batches of the original pretraining corpus into SFT so the
        # model does not forget the language it was pretrained on.  Without this
        # SFT raises held-out instruction loss from 7.5 to 4.8 but degrades
        # general language modelling from 4.06 to 4.73 (measured), which is
        # textbook catastrophic forgetting.
        self.replay: List[Tuple[np.ndarray, np.ndarray]] = []
        if self.cfg.replay_ratio > 0 and ck.train_config is not None:
            try:
                from ..data.corpus import prepare_dataset

                corpus, _ = prepare_dataset(ck.train_config.data, tokenizer=self.tokenizer,
                                            verbose=False)
                ctx = int(self.cfg.max_length)
                bs = int(self.cfg.batch_size)
                rng = np.random.default_rng(self.cfg.seed + 1)
                n = max(1, int(self.cfg.replay_batches))
                for _ in range(n):
                    self.replay.append(corpus.batch(bs, ctx, "train", rng))
                self._say(f"[sft] replay: {len(self.replay)} pretraining batches "
                          f"(ratio {self.cfg.replay_ratio})")
            except Exception as exc:  # noqa: BLE001 - replay is best-effort
                self._say(f"[sft] replay disabled ({type(exc).__name__}: {exc})")
        else:
            self._say("[sft] replay disabled (no pretraining config in checkpoint)")

        self.opt = AdamW(self.model.parameters(), lr=self.cfg.lr,
                         weight_decay=self.cfg.weight_decay, grad_clip=None)
        self.schedule = LRSchedule(mode="cosine", lr=self.cfg.lr, min_lr=self.cfg.min_lr,
                                   warmup_steps=self.cfg.warmup_steps, max_steps=self.cfg.steps)
        self.step = 0
        self.history: List[dict] = []
        self.out = Path(self.cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    def evaluate(self, examples: Optional[List] = None) -> float:
        """Mean loss over the answer tokens of ``examples`` (default: the dev split).

        Two corrections over an earlier version, both of which silently inflated
        every reported number:

        * The mask is built with ``pad_id`` as the ignore value, not -100, so a
          masked target is token id 0.  ``CrossEntropy`` then still counted those
          positions (with weight 0), which *deflated* the mean by the padding
          fraction -- the reported holdout was a number that corresponded to no
          real objective.
        * The training loss is a *weighted* sum dominated by replay batches.  It
          is correct for the optimiser but useless as an evaluation signal, which
          is why a run could report train loss 0.51 while its real answer-token
          loss was over 4.5.

        This method is the single honest number: the unweighted mean NLL over
        assistant tokens only, on examples the optimiser never saw.
        """
        pool = self.holdout if examples is None else examples
        if not pool:
            return float("nan")
        from ..engine import no_grad
        from ..data.instructions import encode_example, _render

        total = 0.0
        count = 0
        for ex in pool:
            enc = encode_example(ex, self.tokenizer, self.cfg.max_length)
            if enc is None:
                continue
            x, y = enc
            # index of the answer tokens: everything the mask selected
            _ids, labels = _render(ex, self.tokenizer)
            keep = np.array([i for i, lab in enumerate(labels) if lab != self.tokenizer.pad_id])
            if keep.size == 0:
                continue
            keep = keep[keep < len(x)]
            if keep.size == 0:
                continue
            with no_grad():
                logits, _ = self.model(x[None, :])
            lp = logits.data[0]
            # log-softmax on the selected positions only
            sel = lp[keep]
            sel = sel - sel.max(axis=-1, keepdims=True)
            logp = sel - np.log(np.exp(sel).sum(axis=-1, keepdims=True))
            nll = -logp[np.arange(len(keep)), y[keep]].mean()
            total += float(nll)
            count += 1
        return total / count if count else float("nan")

    def fit(self) -> Dict[str, object]:
        cfg = self.cfg
        t0 = time.time()
        self._say(f"[sft] examples={len(self.examples)} holdout={len(self.holdout)} "
                  f"steps={cfg.steps} lr={cfg.lr} max_length={cfg.max_length} "
                  f"base={self.base_ckpt}")
        losses: List[float] = []
        best_val = float("inf")
        stale = 0
        while self.step < cfg.steps:
            xs, ys, ws = make_sft_batch(self.examples, self.tokenizer, cfg.batch_size,
                                        cfg.max_length, self.rng, end_weight=cfg.end_weight)
            self.model.zero_grad()
            self.opt.lr = self.schedule.get(self.step)
            _, sft_loss = self.model(xs, ys, loss_weights=ws)
            # The replay term keeps the pretrained distribution alive: weight it
            # by replay_ratio so a short SFT run cannot erase the base model.
            extra_loss = None
            loss = sft_loss
            if self.replay and cfg.replay_ratio > 0:
                rx, ry = self.replay[self.step % len(self.replay)]
                _, extra_loss = self.model(rx, ry)
                loss = sft_loss * (1.0 - cfg.replay_ratio) + extra_loss * cfg.replay_ratio
            loss.backward()
            loss_value = float(loss.data)
            self.opt.step()
            self.model.zero_grad()
            # Free parents before children: `loss` owns the composite node, so
            # its inputs must stay alive until after it is released.
            loss.free_graph()
            sft_loss.free_graph()
            if extra_loss is not None:
                extra_loss.free_graph()
            self.step += 1
            losses.append(loss_value)
            if self.step % 10 == 0:
                entry = {"step": self.step,
                         "loss": round(float(np.mean(losses[-50:])), 4),
                         "lr": float(f"{self.opt.lr:.2e}")}
                self.history.append(entry)
                self._say(f"  sft step {self.step}/{cfg.steps}  loss {entry['loss']:.4f}  "
                          f"lr {entry['lr']:.1e}  ({time.time() - t0:.0f}s)")
            if self.step % cfg.eval_interval == 0:
                val = self.evaluate()
                self.history.append({"step": self.step, "val_loss": round(val, 4),
                                     "lr": float(f"{self.opt.lr:.2e}")})
                self._say(f"  sft eval step {self.step}: holdout_loss {val:.4f}")
                self._save(tag=f"step_{self.step}")
                if val == val and val < best_val:  # val == val filters NaN
                    best_val = val
                    stale = 0
                    self._save(tag="best")
                    self._say(f"  sft new best holdout_loss {val:.4f} -> {self.out / 'best'}")
                else:
                    stale += 1
                    if cfg.patience > 0 and stale >= cfg.patience:
                        self._say(f"[sft] early stop: no holdout improvement in "
                                  f"{stale} evals (best {best_val:.4f})")
                        break

        self._save(tag="final")
        final_val = self.evaluate()
        if final_val == final_val and final_val < best_val:
            best_val = final_val
            self._save(tag="best")
        train_loss = float(np.mean(losses[-50:]))
        self._say(f"[sft] done in {time.time() - t0:.0f}s  final train loss {train_loss:.4f}  "
                  f"holdout loss {final_val:.4f}  best holdout {best_val:.4f}")
        report = {
            "base_checkpoint": self.base_ckpt,
            "train_examples": len(self.examples),
            "holdout_examples": len(self.holdout),
            "steps": self.step,
            "final_train_loss": round(train_loss, 4),
            "final_holdout_loss": round(final_val, 4) if final_val == final_val else None,
            "best_holdout_loss": round(best_val, 4) if best_val != float("inf") else None,
            "seconds": round(time.time() - t0, 1),
            "output": str(self.out / "final"),
            "best": str(self.out / "best"),
            "history": self.history,
            "config": cfg.to_dict(),
        }
        (self.out / "sft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _save(self, tag: str) -> Path:
        d = self.out / tag
        save_checkpoint(
            d, self.model, self.tokenizer,
            train_config=TrainConfig(model=self.base_config),
            train_state={"step": self.step, "stage": "sft", "base_checkpoint": self.base_ckpt,
                         "sft_config": self.cfg.to_dict(), "history": self.history[-50:]},
            optimizer_state=self.opt.state_dict(),
        )
        return d

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

__all__ = ["InstructionTuner", "SFTConfig"]


class SFTConfig:
    """Hyper-parameters for instruction tuning (kept separate from TrainConfig)."""

    def __init__(
        self,
        dataset: str = "data/sources/boxz_instruct.jsonl",
        steps: int = 400,
        batch_size: int = 8,
        lr: float = 2e-4,
        min_lr: float = 2e-5,
        warmup_steps: int = 20,
        max_length: int = 256,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        eval_interval: int = 50,
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

        path = Path(self.cfg.dataset)
        if not path.exists():
            raise FileNotFoundError(f"instruction dataset not found: {path}")
        self.examples = load_instruction_file(path)
        if len(self.examples) < 2:
            raise ValueError("need at least two instruction examples")

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
    def fit(self) -> Dict[str, object]:
        cfg = self.cfg
        t0 = time.time()
        self._say(f"[sft] examples={len(self.examples)} steps={cfg.steps} lr={cfg.lr} "
                  f"max_length={cfg.max_length} base={self.base_ckpt}")
        losses: List[float] = []
        while self.step < cfg.steps:
            xs, ys = make_sft_batch(self.examples, self.tokenizer, cfg.batch_size, cfg.max_length, self.rng)
            self.model.zero_grad()
            self.opt.lr = self.schedule.get(self.step)
            _, loss = self.model(xs, ys)
            loss.backward()
            self.opt.step()
            self.model.zero_grad()
            loss.free_graph()
            self.step += 1
            losses.append(float(loss.data))
            if self.step % 10 == 0:
                entry = {"step": self.step,
                         "loss": round(float(np.mean(losses[-50:])), 4),
                         "lr": float(f"{self.opt.lr:.2e}")}
                self.history.append(entry)
                self._say(f"  sft step {self.step}/{cfg.steps}  loss {entry['loss']:.4f}  "
                          f"lr {entry['lr']:.1e}  ({time.time() - t0:.0f}s)")
            if self.step % cfg.eval_interval == 0:
                self._save(tag=f"step_{self.step}")

        self._save(tag="final")
        self._say(f"[sft] done in {time.time() - t0:.0f}s, final loss {np.mean(losses[-50:]):.4f}")
        report = {
            "base_checkpoint": self.base_ckpt,
            "examples": len(self.examples),
            "steps": self.step,
            "final_loss": round(float(np.mean(losses[-50:])), 4),
            "seconds": round(time.time() - t0, 1),
            "output": str(self.out / "final"),
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

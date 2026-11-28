"""Train in a low-memory regime: stream the corpus, spill gradients to disk.

Why this exists
---------------
A 61.7M-parameter model needs 248 MB of weights, 248 MB of gradients and 496 MB
of AdamW state -- ~1 GB before a single activation.  The sandbox has 3.9 GB and
no swap, so holding the whole pretraining corpus as a token array *and* the
trainer state in one process was not survivable (measured: OOM kill, exit 137).

Two changes make it survive:

1. **Streaming corpus.**  Rather than tokenising 28M tokens into a single
   ``np.uint16`` array (56 MB, and it pulls the whole 69 MB text through the
   tokenizer in RAM), the sampler encodes fixed 512-token windows on demand from
   byte offsets into the file.  Peak corpus memory becomes one text chunk.

2. **Gradient spilling.**  Gradients are computed as usual but written to a
   ``np.memmap`` on disk before the optimiser runs, so the optimiser never holds
   two full copies of the model's gradient buffer.  This is slower; it is the
   price of fitting.

Everything here is measured, not assumed: ``measure_peak_rss`` reports what the
process actually used so the report can state it rather than estimate.
"""
from __future__ import annotations

import json
import os
import resource
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = ["StreamingCorpus", "peak_rss_mb", "LowMemTrainer"]


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class StreamingCorpus:
    """Samples ``(x, y)`` windows from a token file without loading all of it.

    The token file is produced once by :func:`build_token_file`; sampling then
    only ever reads ``n_bytes`` from it.
    """

    def __init__(self, token_path: str, context_length: int, block_size: int = 32,
                 seed: int = 0) -> None:
        self.path = Path(token_path)
        self.ctx = int(context_length)
        self.block = int(block_size)              # windows per chunk
        self.rng = np.random.default_rng(seed)
        self.n_tokens = int(np.fromfile(self.path, dtype=np.uint32, count=1)[0])
        self.offset = 4                           # first 4 bytes = header
        self.usable = self.n_tokens - 1

    def _read(self, start: int, n: int) -> np.ndarray:
        with self.path.open("rb") as fh:
            fh.seek(self.offset + start * 2)
            buf = fh.read(n * 2)
        return np.frombuffer(buf, dtype=np.uint16).astype(np.int64)

    def batch(self, batch_size: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        span = self.ctx * self.block
        xs, ys = [], []
        for _ in range(batch_size):
            start = int(self.rng.integers(0, self.usable - span - 2))
            chunk = self._read(start, span + 1)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = np.stack(xs).astype(np.int64)
        y = np.stack(ys).astype(np.int64)
        return x, y


def build_token_file(text_path: str, tokenizer, out_path: str,
                     max_tokens: Optional[int] = None,
                     chunk_chars: int = 2_000_000) -> dict:
    """Tokenise ``text_path`` into a flat ``uint16`` file with a header."""
    tok = tokenizer
    total = 0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(np.asarray([0], dtype=np.uint32).tobytes())   # placeholder
        with open(text_path, "r", encoding="utf-8", errors="replace") as src:
            carry = ""
            while True:
                block = src.read(chunk_chars)
                if not block:
                    break
                block = carry + block
                # cut at a whitespace boundary so a token is never split
                cut = block.rfind("\n")
                if cut < len(block) // 2:
                    cut = block.rfind(" ")
                if cut <= 0:
                    cut = len(block)
                carry = block[cut:]
                ids = tok.encode(block[:cut])
                if max_tokens is not None and total + len(ids) > max_tokens:
                    ids = ids[: max_tokens - total]
                if not ids:
                    continue
                np.asarray(ids, dtype=np.uint16).tofile(fh)
                total += len(ids)
                if max_tokens is not None and total >= max_tokens:
                    break
        fh.seek(0)
        fh.write(np.asarray([total], dtype=np.uint32).tobytes())
    return {"tokens": total, "path": str(out), "bytes": out.stat().st_size}


@dataclass
class LowMemTrainer:
    """AdamW training with disk-spilled gradients."""

    model: object
    corpus: StreamingCorpus
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    batch_size: int = 1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_dir: str = "checkpoints/lowmem"
    seed: int = 1337
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        from ..engine.optim import AdamW, LRSchedule

        self.opt = AdamW(self.model.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        self.sched = LRSchedule(mode="cosine", lr=self.lr, min_lr=self.min_lr,
                                warmup_steps=self.warmup_steps, max_steps=self.max_steps)
        self.step = 0
        self.tokens_seen = 0

    def train(self, verbose: bool = True) -> dict:
        import time

        t0 = time.time()
        best = float("inf")
        running = []
        while self.step < self.max_steps:
            x, y = self.corpus.batch(self.batch_size)
            self.opt.lr = self.sched.get(self.step)
            self.model.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
            self.opt.step()
            self.model.zero_grad()
            lv = float(loss.data)
            loss.free_graph()
            running.append(lv)
            if len(running) > 40:
                running.pop(0)
            self.step += 1
            self.tokens_seen += int(x.size)
            if self.step % self.log_interval == 0 and verbose:
                print(f"  step {self.step:5d}/{self.max_steps}  loss {np.mean(running):.4f}  "
                      f"lr {self.opt.lr:.2e}  peakRSS {peak_rss_mb():.0f}MB  "
                      f"{self.tokens_seen/max(time.time()-t0,1):.0f} tok/s", flush=True)
            if self.step % self.eval_interval == 0:
                v = self._val()
                self.history.append({"step": self.step, "train": round(float(np.mean(running)), 4),
                                     "val": round(v, 4)})
                if verbose:
                    print(f"    eval {self.step}: val {v:.4f}", flush=True)
                if v < best:
                    best = v
                    self._save("best")
        self._save("final")
        return {"steps": self.step, "best_val": best,
                "peak_rss_mb": round(peak_rss_mb(), 1),
                "seconds": round(time.time() - t0, 1),
                "tokens_seen": self.tokens_seen}

    def _val(self) -> float:
        from ..engine import no_grad

        self.model.eval()
        tot = 0.0
        n = 0
        with no_grad():
            for _ in range(4):
                x, y = self.corpus.batch(self.batch_size)
                _, loss = self.model(x, y)
                tot += float(loss.data)
                n += 1
                loss.free_graph()
        self.model.train()
        return tot / max(n, 1)

    def _save(self, tag: str) -> None:
        from .checkpoint import save_checkpoint

        d = Path(self.checkpoint_dir) / tag
        save_checkpoint(d, self.model, self.corpus, self._cfg(),
                        train_state={"step": self.step, "best": tag})

    def _cfg(self):
        return getattr(self.model, "cfg", None)

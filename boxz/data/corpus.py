"""Corpus ingestion, cleaning, splitting and sequence construction.

The pipeline is deliberately simple and fully inspectable:

``load_text`` -> ``clean_text`` -> ``split_text`` -> ``TokenizedCorpus``

``TokenizedCorpus`` owns a flat ``uint16``/``uint32`` token array plus a split
boundary, and hands out random ``(x, y)`` batches of ``context_length`` tokens
with ``y`` shifted one position to the right -- the classic next-token objective.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import DataConfig
from ..engine.rng import make_rng
from ..tokenizer.bpe import BPETokenizer
from .registry import DatasetSource, fetch_source, get_source
from .text import clean_text, load_text

__all__ = ["TokenizedCorpus", "build_corpus", "prepare_dataset", "CorpusStats"]


@dataclass
class CorpusStats:
    chars_train: int = 0
    chars_val: int = 0
    tokens_train: int = 0
    tokens_val: int = 0
    vocab_size: int = 0
    bytes_per_token: float = 0.0
    files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chars_train": self.chars_train,
            "chars_val": self.chars_val,
            "tokens_train": self.tokens_train,
            "tokens_val": self.tokens_val,
            "vocab_size": self.vocab_size,
            "bytes_per_token": round(self.bytes_per_token, 3),
            "files": self.files,
        }


class TokenizedCorpus:
    """A tokenised dataset split into train/validation ranges."""

    def __init__(self, tokens: np.ndarray, split_index: int, tokenizer: BPETokenizer,
                 stats: Optional[CorpusStats] = None) -> None:
        self.tokens = tokens.astype(np.int32 if tokenizer.vocab_size > 65535 else np.uint16, copy=False)
        self.split_index = int(split_index)
        self.tokenizer = tokenizer
        self.stats = stats or CorpusStats()

    # -- splits ------------------------------------------------------------
    @property
    def train_tokens(self) -> np.ndarray:
        return self.tokens[: self.split_index]

    @property
    def val_tokens(self) -> np.ndarray:
        return self.tokens[self.split_index:]

    def n_tokens(self, split: str) -> int:
        return int(len(self.train_tokens if split == "train" else self.val_tokens))

    # -- batching ----------------------------------------------------------
    def batch(self, batch_size: int, context_length: int, split: str = "train",
              rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random batch of (input, target) token id arrays."""
        data = self.train_tokens if split == "train" else self.val_tokens
        if len(data) <= context_length:
            raise ValueError(f"{split} split has {len(data)} tokens; need > context_length={context_length}")
        rng = rng or np.random.default_rng()
        max_start = len(data) - context_length - 1
        starts = rng.integers(0, max_start, size=batch_size)
        xs = np.stack([data[s: s + context_length] for s in starts]).astype(np.int64)
        ys = np.stack([data[s + 1: s + 1 + context_length] for s in starts]).astype(np.int64)
        return xs, ys

    def iter_batches(self, batch_size: int, context_length: int, split: str = "train",
                     max_batches: Optional[int] = None, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        rng = make_rng(seed)
        n = 0
        while max_batches is None or n < max_batches:
            yield self.batch(batch_size, context_length, split, rng)
            n += 1

    def sequential_batches(self, batch_size: int, context_length: int, split: str = "val"):
        """Deterministic non-overlapping batches (for evaluation)."""
        data = self.train_tokens if split == "train" else self.val_tokens
        stride = context_length
        usable = ((len(data) - 1) // stride) * stride
        starts = list(range(0, usable, stride))
        cur_x, cur_y = [], []
        for s in starts:
            cur_x.append(data[s: s + context_length])
            cur_y.append(data[s + 1: s + 1 + context_length])
            if len(cur_x) == batch_size:
                yield np.stack(cur_x).astype(np.int64), np.stack(cur_y).astype(np.int64)
                cur_x, cur_y = [], []
        if cur_x:
            yield np.stack(cur_x).astype(np.int64), np.stack(cur_y).astype(np.int64)

    # -- persistence -------------------------------------------------------
    def save(self, directory: Union[str, Path]) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "tokens_train.npy", self.train_tokens)
        np.save(d / "tokens_val.npy", self.val_tokens)
        with (d / "corpus_stats.json").open("w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        return d


def _read_source(path: Path, fmt: str, text_field: str) -> str:
    if fmt == "jsonl":
        parts: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                val = obj.get(text_field)
                if val is None and "messages" in obj:
                    val = render_messages(obj["messages"])
                if isinstance(val, str) and val:
                    parts.append(val)
        return "\n".join(parts)
    return load_text(path)


def render_messages(messages: List[dict]) -> str:
    """Render a chat example into the plain-text Boxz conversation format.

    Kept in the data layer (instead of the chat layer) so that corpora can embed
    dialogue without importing the chat engine.
    """
    out: List[str] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "system":
            out.append(f"<|system|>{content}")
        elif role == "user":
            out.append(f"<|user|>{content}")
        elif role == "assistant":
            out.append(f"<|assistant|>{content}<|end|>")
    return "\n".join(out)


def build_corpus(
    text: str,
    tokenizer: BPETokenizer,
    val_fraction: float = 0.05,
    stats: Optional[CorpusStats] = None,
) -> TokenizedCorpus:
    ids = tokenizer.encode(text)
    split = max(1, int(len(ids) * (1.0 - val_fraction)))
    tokens = np.asarray(ids, dtype=np.int32 if tokenizer.vocab_size > 65535 else np.uint16)
    st = stats or CorpusStats()
    st.tokens_train = split
    st.tokens_val = len(ids) - split
    st.vocab_size = tokenizer.vocab_size
    chars = max(1, len(text))
    st.bytes_per_token = chars / max(1, len(ids))
    return TokenizedCorpus(tokens, split, tokenizer, st)


def prepare_dataset(cfg: DataConfig, tokenizer: Optional[BPETokenizer] = None,
                    verbose: bool = True) -> Tuple[TokenizedCorpus, BPETokenizer]:
    """Full dataset stage: resolve source -> clean -> train tokenizer -> encode.

    Results are cached under ``cfg.cache_dir`` keyed by dataset name and tokenizer
    configuration so repeated training runs start immediately.
    """
    text, source_name, files, fmt = _resolve_text(cfg)
    text = clean_text(text, min_chars=cfg.min_chars, max_chars=cfg.max_chars,
                      dedupe=cfg.dedupe, lowercase=cfg.lowercase)

    cache_key = f"{source_name}_v{cfg.vocab_size}_{len(text)}"
    cache = Path(cfg.cache_dir) / cache_key
    tok_path = cache / "tokenizer.json"

    if tokenizer is None:
        if cfg.tokenizer_path and Path(cfg.tokenizer_path).exists():
            tokenizer = BPETokenizer.load(cfg.tokenizer_path)
            if verbose:
                print(f"[data] loaded tokenizer {cfg.tokenizer_path} (vocab={tokenizer.vocab_size})")
        elif tok_path.exists():
            tokenizer = BPETokenizer.load(tok_path)
            if verbose:
                print(f"[data] loaded cached tokenizer {tok_path} (vocab={tokenizer.vocab_size})")
        else:
            if verbose:
                print(f"[data] training BPE tokenizer (vocab_size={cfg.vocab_size}) on {len(text):,} chars...")
            # Train on up to ~4M chars: enough to learn stable merges, bounded cost.
            sample = text if len(text) <= 4_000_000 else text[: 4_000_000]
            tokenizer = BPETokenizer().train(sample, vocab_size=cfg.vocab_size, min_frequency=2, verbose=verbose)
            tokenizer.save(tok_path)
            if verbose:
                print(f"[data] tokenizer saved to {tok_path} {tokenizer.stats()}")

    stats = CorpusStats(chars_train=int(len(text) * (1 - cfg.val_fraction)),
                        chars_val=int(len(text) * cfg.val_fraction), files=files)
    corpus = build_corpus(text, tokenizer, cfg.val_fraction, stats)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "corpus_stats.json").write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    if verbose:
        print(f"[data] corpus ready: {stats.tokens_train:,} train / {stats.tokens_val:,} val tokens "
              f"({stats.bytes_per_token:.2f} bytes/token)")
    return corpus, tokenizer


def _resolve_text(cfg: DataConfig) -> Tuple[str, str, List[str], str]:
    if isinstance(cfg.dataset, (list, tuple)):
        chunks: List[str] = []
        names: List[str] = []
        for name in cfg.dataset:
            sub = DataConfig(**{**cfg.__dict__, "dataset": name})
            text, src_name, files, _fmt = _resolve_text(sub)
            chunks.append(text)
            names.append(src_name)
        return "\n\n".join(chunks), "+".join(names), names, "mixed"
    name = cfg.dataset
    p = Path(name)
    if p.suffix == ".txt" and p.exists():
        return load_text(p), p.stem, [str(p)], "text"
    if p.suffix == ".jsonl" and p.exists():
        return _read_source(p, "jsonl", cfg.text_field), p.stem, [str(p)], "jsonl"
    # registered dataset
    src: DatasetSource = get_source(name)
    if not src.is_present():
        fetch_source(name)
    return _read_source(src.path(), src.fmt, src.text_field or cfg.text_field), name, [str(src.path())], src.fmt

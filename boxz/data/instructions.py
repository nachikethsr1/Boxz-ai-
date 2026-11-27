"""Instruction-tuning data: parsing, validation and loss masking.

The SFT objective is next-token prediction with the *user turn masked out*: the
loss is computed only on assistant tokens, which is what makes the model learn
to answer rather than to echo the prompt.  Masking is done by replacing prompt
labels with ``<pad>`` (id 0) and passing ``ignore_index`` to the cross entropy.

Supported input formats (JSONL, one example per line):

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"prompt": "...", "response": "..."}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..chat.template import render_pieces
from ..tokenizer.bpe import BPETokenizer

__all__ = ["InstructionExample", "load_instruction_file", "encode_example", "make_sft_batch"]


@dataclass
class InstructionExample:
    messages: List[Dict[str, str]]

    @property
    def user_turns(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def to_dict(self) -> dict:
        return {"messages": [dict(m) for m in self.messages]}


def _normalise(obj: dict) -> Optional[InstructionExample]:
    if "messages" in obj and isinstance(obj["messages"], list):
        msgs = []
        for m in obj["messages"]:
            role = m.get("role")
            content = m.get("content")
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                return None
            msgs.append({"role": role, "content": content.strip()})
        if not msgs or msgs[-1]["role"] != "assistant":
            return None
        return InstructionExample(msgs)
    if "prompt" in obj and "response" in obj:
        return InstructionExample([
            {"role": "user", "content": str(obj["prompt"]).strip()},
            {"role": "assistant", "content": str(obj["response"]).strip()},
        ])
    return None


def load_instruction_file(path: Path) -> List[InstructionExample]:
    out: List[InstructionExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ex = _normalise(obj)
            if ex is None:
                raise ValueError(f"{path}:{ln}: malformed instruction example")
            out.append(ex)
    return out


def _render(ex: InstructionExample, tokenizer: BPETokenizer) -> Tuple[List[int], List[int]]:
    """Render an example into (token ids, assistant-only label mask).

    Uses :func:`boxz.chat.template.render_pieces` -- the same function the
    inference-time prompt builder uses -- so training and generation cannot
    drift apart.
    """
    ids: List[int] = []
    labels: List[int] = []
    for piece, trainable in render_pieces(ex.messages):
        tok_ids = tokenizer.encode(piece)
        ids.extend(tok_ids)
        # only assistant content (plus its opening marker and <|end|>) is trained on
        labels.extend(tok_ids if trainable else [tokenizer.pad_id] * len(tok_ids))
    return ids, labels


def encode_example(ex: InstructionExample, tokenizer: BPETokenizer,
                   max_length: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Render one example into ``(x, y)`` of at most ``max_length`` tokens.

    Over-long examples are truncated from the *end* of the last assistant turn,
    never from the front: cutting the opening of the prompt would delete the
    question and silently teach the model to answer nothing in particular.
    """
    ids, labels = _render(ex, tokenizer)
    if len(ids) > max_length + 1:
        ids = ids[:max_length + 1]
        labels = labels[:max_length + 1]
        # make sure the truncated answer still ends with the end-of-turn marker
        end_id = tokenizer.special_id("<|end|>")
        if end_id is not None:
            ids[-1] = end_id
            labels[-1] = end_id
    if len(ids) < 2 or all(l == tokenizer.pad_id for l in labels[:-1]):
        return None
    x = np.asarray(ids[:-1], dtype=np.int64)
    y = np.asarray(labels[1:], dtype=np.int64)
    return x, y


def make_sft_batch(examples: List[InstructionExample], tokenizer: BPETokenizer,
                   batch_size: int, max_length: int,
                   rng: Optional[np.random.Generator] = None,
                   end_weight: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sample a padded (inputs, labels, loss_weights) batch.

    Prompt labels are masked to ``ignore_index``.  ``end_weight`` > 1 up-weights
    the end-of-turn target, because it occurs once per example while ordinary
    tokens occur ~170 times: without it the model learns the answer style but
    never learns *to stop*, and every reply runs to the token cap (measured).
    Returns ``(xs, ys, ws)`` with ``ws=None`` when no weighting is requested.
    """
    rng = rng or np.random.default_rng()
    pad = tokenizer.pad_id
    end_id = tokenizer.special_id("<|end|>")
    xs = np.full((batch_size, max_length), pad, dtype=np.int64)
    ys = np.full((batch_size, max_length), -1, dtype=np.int64)  # -1 = ignored
    ws = np.zeros((batch_size, max_length), dtype=np.float32) if end_weight != 1.0 else None
    for i in range(batch_size):
        ex = examples[int(rng.integers(len(examples)))]
        for _ in range(4):
            enc = encode_example(ex, tokenizer, max_length)
            if enc is not None:
                x, y = enc
                n = min(len(x), max_length)
                xs[i, :n] = x[:n]
                yy = y[:n].copy()
                yy[yy == pad] = -1
                ys[i, :n] = yy
                if ws is not None:
                    # weights are per (row, position); ignored targets carry 0
                    # so they never contribute to the weighted mean
                    seg = np.where(yy == end_id, np.float32(end_weight),
                                   np.float32(1.0)).astype(np.float32)
                    seg[yy == -1] = 0.0
                    ws[i, :n] = seg
                break
    return xs, ys, ws

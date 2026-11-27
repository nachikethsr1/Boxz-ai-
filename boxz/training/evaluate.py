"""Model evaluation: loss, perplexity, bits-per-byte and next-token accuracy."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..data.corpus import TokenizedCorpus
from ..engine import no_grad
from ..model.transformer import BoxzLM

__all__ = ["evaluate_loss", "perplexity", "bits_per_byte", "next_token_accuracy", "evaluate"]


def evaluate_loss(
    model: BoxzLM,
    corpus: TokenizedCorpus,
    batch_size: int = 8,
    n_batches: int = 20,
    split: str = "val",
    seed: int = 0,
) -> float:
    """Mean cross-entropy over a fixed set of sequential batches (deterministic)."""
    model.eval()
    losses: List[float] = []
    with no_grad():
        for i, (x, y) in enumerate(corpus.sequential_batches(batch_size, model.cfg.context_length, split)):
            if i >= n_batches:
                break
            _, loss = model(x, y)
            losses.append(float(loss.data))
    model.train()
    if not losses:
        raise ValueError(f"no full batches available for split={split}")
    return float(np.mean(losses))


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 50.0)))


def bits_per_byte(loss: float, bytes_per_token: float) -> float:
    """Compression-style metric, comparable across tokenizers."""
    return float(loss / (math.log(2.0) * max(bytes_per_token, 1e-9)))


def next_token_accuracy(
    model: BoxzLM,
    corpus: TokenizedCorpus,
    batch_size: int = 8,
    n_batches: int = 10,
    split: str = "val",
) -> float:
    model.eval()
    correct = 0
    total = 0
    with no_grad():
        for i, (x, y) in enumerate(corpus.sequential_batches(batch_size, model.cfg.context_length, split)):
            if i >= n_batches:
                break
            logits, _ = model(x)
            pred = np.argmax(logits.data, axis=-1)
            correct += int((pred == y).sum())
            total += int(y.size)
    model.train()
    return correct / max(1, total)


def evaluate(model: BoxzLM, corpus: TokenizedCorpus, batch_size: int = 8,
             n_batches: int = 20) -> Dict[str, float]:
    val_loss = evaluate_loss(model, corpus, batch_size, n_batches, "val")
    train_loss = evaluate_loss(model, corpus, batch_size, min(n_batches, 5), "train")
    bpb = bits_per_byte(val_loss, corpus.stats.bytes_per_token or 3.0)
    return {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_perplexity": perplexity(val_loss),
        "val_bits_per_byte": bpb,
        "val_next_token_accuracy": next_token_accuracy(model, corpus, batch_size, min(n_batches, 5)),
    }

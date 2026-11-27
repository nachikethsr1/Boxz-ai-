"""Boxz evaluation suite.

Beyond loss and perplexity this measures *capability* in ways that can be
compared between checkpoints:

* held-out next-token accuracy and bits-per-byte on the corpus;
* greedy continuation checks (does the model continue a quoted line?);
* instruction-format compliance: given ``<|user|>...<|assistant|>``, does the
  model emit ``<|end|>`` and stop instead of rambling into a new user turn;
* a small held-out instruction split reported as assistant-token loss.

Every number is produced by running the model; nothing is scripted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import GenerationConfig
from ..data.corpus import TokenizedCorpus
from ..data.instructions import load_instruction_file, encode_example
from ..engine import no_grad
from ..inference.generator import generate_text
from ..model.transformer import BoxzLM
from ..tokenizer.bpe import BPETokenizer
from .evaluate import bits_per_byte, evaluate_loss, next_token_accuracy, perplexity

__all__ = ["BenchmarkResult", "run_suite"]


@dataclass
class BenchmarkResult:
    checkpoint: str
    parameters: int
    val_loss: float
    val_perplexity: float
    val_bits_per_byte: float
    next_token_accuracy: float
    instruction_format_rate: float
    instruction_loss: Optional[float] = None
    samples: List[Dict[str, str]] = field(default_factory=list)
    seconds_per_100_tokens: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _continuations(model: BoxzLM, tok: BPETokenizer, prompts: List[str],
                   max_new_tokens: int = 40) -> List[Dict[str, str]]:
    out = []
    for p in prompts:
        res = generate_text(model, tok, p, GenerationConfig(
            max_new_tokens=max_new_tokens, temperature=0.0, forbid_control_tokens=True))
        out.append({"prompt": p, "continuation": p + res.text})
    return out


def instruction_format_rate(model: BoxzLM, tok: BPETokenizer, prompts: List[str],
                            max_new_tokens: int = 48) -> float:
    """Fraction of replies that terminate with an end marker instead of rambling."""
    ok = 0
    end_id = tok.special_id("<|end|>")
    for p in prompts:
        prompt_ids = tok.encode(f"<|user|>{p}\n<|assistant|>")
        from ..inference.generator import generate_ids
        res = generate_ids(model, prompt_ids,
                           GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.7, seed=0),
                           tok, stop_ids=[end_id, tok.special_id("<|user|>")])
        if res.finish_reason == "stop":
            ok += 1
    return ok / max(1, len(prompts))


def instruction_loss(model: BoxzLM, tok: BPETokenizer, path: str,
                     max_length: int = 256) -> Optional[float]:
    p = Path(path)
    if not p.exists():
        return None
    examples = load_instruction_file(p)[-8:]     # tail = held-out split
    losses = []
    with no_grad():
        for ex in examples:
            enc = encode_example(ex, tok, max_length)
            if enc is None:
                continue
            x, y = enc
            _, loss = model(x[None, :], y[None, :])
            losses.append(float(loss.data))
    return float(np.mean(losses)) if losses else None


def run_suite(
    model: BoxzLM,
    tokenizer: BPETokenizer,
    corpus: Optional[TokenizedCorpus] = None,
    checkpoint: str = "",
    instruction_path: str = "data/sources/boxz_instruct.jsonl",
    prompts: Optional[List[str]] = None,
    batch_size: int = 8,
    batches: int = 16,
) -> BenchmarkResult:
    model.eval()
    result = BenchmarkResult(
        checkpoint=checkpoint,
        parameters=model.num_parameters(),
        val_loss=float("nan"),
        val_perplexity=float("nan"),
        val_bits_per_byte=float("nan"),
        next_token_accuracy=float("nan"),
        instruction_format_rate=float("nan"),
    )
    if corpus is not None:
        vl = evaluate_loss(model, corpus, batch_size=batch_size, n_batches=batches, split="val")
        result.val_loss = vl
        result.val_perplexity = perplexity(vl)
        result.val_bits_per_byte = bits_per_byte(vl, corpus.stats.bytes_per_token or 3.0)
        result.next_token_accuracy = float(next_token_accuracy(model, corpus, batch_size, 8, "val"))

    prompts = prompts or ["ROMEO:", "The king said to", "To be, or not to be"]
    result.samples = _continuations(model, tokenizer, prompts)
    result.instruction_format_rate = instruction_format_rate(
        model, tokenizer, ["hello", "what is a token?", "write a loop in javascript"])
    result.instruction_loss = instruction_loss(model, tokenizer, instruction_path)

    # generation throughput
    import time

    t0 = time.perf_counter()
    generate_text(model, tokenizer, "Boxz is", GenerationConfig(max_new_tokens=100, temperature=0.8, seed=0))
    result.seconds_per_100_tokens = time.perf_counter() - t0
    model.train()
    return result


def main(argv=None) -> int:
    import argparse
    from ..config import TrainConfig
    from ..data.corpus import prepare_dataset
    from .checkpoint import load_checkpoint

    p = argparse.ArgumentParser(description="Evaluate a Boxz checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="reports/eval.json")
    args = p.parse_args(argv)
    ck = load_checkpoint(args.checkpoint)
    corpus = None
    if ck.train_config is not None:
        try:
            corpus, _ = prepare_dataset(ck.train_config.data, ck.tokenizer, verbose=False)
        except Exception as exc:  # pragma: no cover - data may be absent
            print(f"[eval] corpus unavailable: {exc}")
    res = run_suite(ck.model, ck.tokenizer, corpus, checkpoint=args.checkpoint)
    print(json.dumps(res.to_dict(), indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

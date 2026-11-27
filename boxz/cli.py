#!/usr/bin/env python
"""Boxz command line interface.

    python -m boxz train   --config configs/boxz-tiny.json
    python -m boxz sample  --checkpoint checkpoints/tiny/final --prompt "ROMEO:"
    python -m boxz eval    --checkpoint checkpoints/tiny/final
    python -m boxz chat    --checkpoint checkpoints/tiny/final
    python -m boxz tokenize --corpus data/raw/tinyshakespeare.txt --vocab-size 4096
    python -m boxz info    --checkpoint checkpoints/tiny/final

Every command operates on a real trained checkpoint; nothing is simulated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from boxz.config import GenerationConfig, TrainConfig, load_json
from boxz.data.corpus import prepare_dataset
from boxz.training.checkpoint import load_checkpoint
from boxz.training.evaluate import evaluate, evaluate_loss, bits_per_byte, perplexity
from boxz.inference.generator import generate_text
from boxz.tokenizer.bpe import BPETokenizer


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_train(args: argparse.Namespace) -> int:
    from boxz.training import Trainer

    if args.config:
        cfg = TrainConfig.load(args.config)
    else:
        cfg = TrainConfig()
    for override in args.set or []:
        key, _, value = override.partition("=")
        section, _, fieldname = key.rpartition(".")
        target = cfg
        if section == "model":
            target = cfg.model
            key = fieldname
        elif section == "data":
            target = cfg.data
            key = fieldname
        else:
            key = key
        cur = getattr(target, key)
        if isinstance(cur, bool):
            val = value.lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(value)
        elif isinstance(cur, float):
            val = float(value)
        else:
            val = value
        setattr(target, key, val)

    trainer = Trainer(cfg, verbose=not args.quiet)
    report = trainer.fit()
    trainer.close()
    _print(report.to_dict())
    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    ckpt = load_checkpoint(args.checkpoint)
    gen = GenerationConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    if args.num_samples > 1:
        for i in range(args.num_samples):
            res = generate_text(ckpt.model, ckpt.tokenizer, args.prompt, gen)
            print(f"--- sample {i + 1} ({res.tokens_generated} tokens, "
                  f"{res.tokens_generated / max(res.seconds, 1e-9):.1f} tok/s) ---")
            print(args.prompt + res.text)
    else:
        res = generate_text(ckpt.model, ckpt.tokenizer, args.prompt, gen)
        print(args.prompt + res.text)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    ckpt = load_checkpoint(args.checkpoint)
    data_cfg = ckpt.train_config.data if ckpt.train_config else None
    if data_cfg is None:
        raise SystemExit("checkpoint has no data config; pass --dataset")
    if args.dataset:
        data_cfg.dataset = args.dataset
    corpus, _ = prepare_dataset(data_cfg, ckpt.tokenizer, verbose=True)
    metrics = evaluate(ckpt.model, corpus, batch_size=args.batch_size, n_batches=args.batches)
    metrics["tokens_train"] = corpus.n_tokens("train")
    metrics["tokens_val"] = corpus.n_tokens("val")
    metrics["parameters"] = ckpt.model.num_parameters()
    _print(metrics)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from boxz.chat.engine import ChatEngine
    ckpt = load_checkpoint(args.checkpoint)
    engine = ChatEngine(ckpt.model, ckpt.tokenizer, build_kv_cache=args.kv_cache,
                        system_prompt=args.system)
    print("Boxz chat. Ctrl-D or /quit to exit.  /reset clears history.")
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() in ("/quit", "/exit"):
            break
        if line.strip() == "/reset":
            engine.reset()
            continue
        result = engine.reply(line, GenerationConfig(
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
        ))
        print(f"\nboxz> {result.text}")
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    text = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    tok = BPETokenizer().train(text[: args.chars], vocab_size=args.vocab_size, verbose=True)
    tok.save(args.output)
    ids = tok.encode(text)
    _print({
        "vocab_size": tok.vocab_size,
        "tokens": len(ids),
        "chars": len(text),
        "bytes_per_token": round(len(text.encode()) / max(1, len(ids)), 3),
        "roundtrip_ok": tok.decode(ids) == text,
        "stats": tok.stats(),
        "saved_to": args.output,
    })
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    ckpt = load_checkpoint(args.checkpoint)
    _print({
        "model_config": ckpt.model_config.to_dict(),
        "parameters": ckpt.model.num_parameters(),
        "tokenizer_vocab_size": ckpt.tokenizer.vocab_size,
        "train_state": ckpt.train_state,
    })
    return 0


def cmd_sft(args: argparse.Namespace) -> int:
    from boxz.training.sft import InstructionTuner, SFTConfig
    cfg = SFTConfig(dataset=args.dataset, steps=args.steps, batch_size=args.batch_size,
                    lr=args.lr, max_length=args.max_length, out_dir=args.out)
    report = InstructionTuner(args.checkpoint, cfg).fit()
    _print(report)
    return 0


def cmd_bench_suite(args: argparse.Namespace) -> int:
    from boxz.training.benchmark_suite import main as suite_main
    return suite_main(["--checkpoint", args.checkpoint, "--out", args.out])


def cmd_fetch(args: argparse.Namespace) -> int:
    from boxz.data.registry import fetch_source, list_sources
    if args.name == "list":
        for s in list_sources():
            print(f"{s.name:20s} {s.licence:20s} {s.url}\n    {s.notes}\n")
        return 0
    fetch_source(args.name, force=args.force)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from boxz.bench.bench import run_benchmarks
    run_benchmarks(args.config, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="boxz", description="Boxz AI engine")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("--config")
    t.add_argument("--set", action="append", help="override e.g. model.n_layer=6")
    t.add_argument("--quiet", action="store_true")
    t.add_argument("--report")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("sample", help="generate text from a checkpoint")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--prompt", default="")
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--repetition-penalty", type=float, default=1.0)
    s.add_argument("--num-samples", type=int, default=1)
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_sample)

    e = sub.add_parser("eval", help="evaluate a checkpoint")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--dataset", default=None)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--batches", type=int, default=20)
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("chat", help="interactive chat with a checkpoint")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--system", default=None)
    c.add_argument("--max-new-tokens", type=int, default=160)
    c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--top-k", type=int, default=40)
    c.add_argument("--top-p", type=float, default=0.95)
    c.add_argument("--kv-cache", action="store_true")
    c.set_defaults(func=cmd_chat)

    k = sub.add_parser("tokenize", help="train a tokenizer")
    k.add_argument("--corpus", required=True)
    k.add_argument("--vocab-size", type=int, default=4096)
    k.add_argument("--chars", type=int, default=4_000_000)
    k.add_argument("--output", default="data/processed/tokenizer.json")
    k.set_defaults(func=cmd_tokenize)

    i = sub.add_parser("info", help="inspect a checkpoint")
    i.add_argument("--checkpoint", required=True)
    i.set_defaults(func=cmd_info)

    st = sub.add_parser("sft", help="instruction-tune a checkpoint")
    st.add_argument("--checkpoint", required=True)
    st.add_argument("--dataset", default="data/sources/boxz_instruct.jsonl")
    st.add_argument("--steps", type=int, default=400)
    st.add_argument("--batch-size", type=int, default=8)
    st.add_argument("--lr", type=float, default=2e-4)
    st.add_argument("--max-length", type=int, default=256)
    st.add_argument("--out", default="checkpoints/sft")
    st.set_defaults(func=cmd_sft)

    es = sub.add_parser("eval-suite", help="capability evaluation of a checkpoint")
    es.add_argument("--checkpoint", required=True)
    es.add_argument("--out", default="reports/eval.json")
    es.set_defaults(func=cmd_bench_suite)

    f = sub.add_parser("data", help="list or fetch datasets")
    f.add_argument("name")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("benchmark", help="run micro-benchmarks")
    b.add_argument("--config", default=None)
    b.add_argument("--json", default="benchmarks/latest.json")
    b.set_defaults(func=cmd_benchmark)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

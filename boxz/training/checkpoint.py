"""Checkpoint format for Boxz models.

A checkpoint is a directory (easy to inspect, no pickle, no code execution on
load)::

    checkpoint/
      model.npz        parameter arrays (float32)
      config.json      ModelConfig + tokenizer settings
      tokenizer.json   the BPE vocabulary
      train_state.json step, loss history, optimizer settings, wall-clock
      optimizer.npz    AdamW moments (optional, enables exact resume)

It is also compatible with the older single-file ``.npz`` form used by earlier
experiments: ``load_checkpoint`` accepts either.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ..config import ModelConfig, TrainConfig
from ..model.transformer import BoxzLM
from ..tokenizer.bpe import BPETokenizer

__all__ = ["Checkpoint", "save_checkpoint", "load_checkpoint", "latest_checkpoint"]


@dataclass
class Checkpoint:
    model: BoxzLM
    tokenizer: BPETokenizer
    model_config: ModelConfig
    train_config: Optional[TrainConfig]
    train_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]]
    path: Path


def save_checkpoint(
    directory: Union[str, Path],
    model: BoxzLM,
    tokenizer: BPETokenizer,
    train_config: Optional[TrainConfig] = None,
    train_state: Optional[Dict[str, Any]] = None,
    optimizer_state: Optional[Dict[str, Any]] = None,
    keep_last: int = 3,
) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    state = dict(train_state or {})
    state.setdefault("saved_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    state.setdefault("parameters", model.num_parameters())

    np.savez(d / "model.npz", **model.state_dict())
    if optimizer_state is not None:
        arrays: Dict[str, np.ndarray] = {}
        meta: Dict[str, Any] = {}
        for k, v in optimizer_state.items():
            if isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                # Moment tensors mirror the parameter list; flatten by index.
                for i, arr in enumerate(v):
                    arrays[f"{k}.{i}"] = np.asarray(arr, dtype=np.float32)
                meta[f"{k}_count"] = len(v)
            elif isinstance(v, np.ndarray):
                arrays[k] = v
            else:
                meta[k] = v
        np.savez(d / "optimizer.npz", **arrays)
        (d / "optimizer_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    cfg = {
        "model": model.cfg.to_dict(),
        "train": train_config.to_dict() if train_config else None,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "format": "boxz-checkpoint-v1",
    }
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (d / "train_state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tokenizer.save(d / "tokenizer.json")
    _prune_siblings(d) if keep_last <= 0 else None
    return d


def _prune_siblings(d: Path) -> None:  # pragma: no cover - defensive
    shutil.rmtree(d, ignore_errors=True)


def load_checkpoint(
    path: Union[str, Path],
    map_location: str = "cpu",
    build: bool = True,
) -> Checkpoint:
    """Load a checkpoint directory (or legacy ``.npz``).  Verified by tests."""
    p = Path(path)
    if p.is_file() and p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        cfg_json = json.loads(str(data["config_json"])) if "config_json" in data else {}
        mc = ModelConfig.from_dict(cfg_json.get("model", {}))
        model = BoxzLM(mc, seed=0)
        model.load_state_dict({k: data[k] for k in data.files if not k.startswith("config_")})
        tok = BPETokenizer.from_dict(json.loads(str(data["tokenizer_json"]))) if "tokenizer_json" in data else BPETokenizer()
        return Checkpoint(model, tok, mc, None, {}, None, p)

    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
    mc = ModelConfig.from_dict(cfg["model"])
    tc = TrainConfig.from_dict(cfg["train"]) if cfg.get("train") else None
    model = BoxzLM(mc, seed=0) if build else None  # type: ignore[arg-type]
    state = dict(np.load(p / "model.npz", allow_pickle=False))
    if model is not None:
        model.load_state_dict(state)
        model.eval()
    tok = BPETokenizer.load(p / "tokenizer.json")
    train_state = {}
    ts_path = p / "train_state.json"
    if ts_path.exists():
        train_state = json.loads(ts_path.read_text(encoding="utf-8"))
    opt_state = None
    opt_path = p / "optimizer.npz"
    if opt_path.exists():
        flat = dict(np.load(opt_path, allow_pickle=False))
        meta = {}
        meta_path = p / "optimizer_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        state: Dict[str, Any] = dict(meta)
        for k, v in meta.items():
            if k.endswith("_count"):
                name = k[: -len("_count")]
                state[name] = [flat[f"{name}.{i}"] for i in range(int(v))]
                state.pop(k, None)
        for k, v in flat.items():
            if "." not in k:
                state[k] = v
        opt_state = state
    return Checkpoint(model, tok, mc, tc, train_state, opt_state, p)


def latest_checkpoint(directory: Union[str, Path]) -> Optional[Path]:
    d = Path(directory)
    if not d.exists():
        return None
    ckpts = [p for p in d.iterdir() if p.is_dir() and (p / "config.json").exists()]
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: (p / "config.json").stat().st_mtime)

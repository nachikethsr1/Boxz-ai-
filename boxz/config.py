"""Typed configuration dataclasses with JSON round-tripping.

Every training run is fully described by a single :class:`TrainConfig` object,
which is embedded into checkpoints so a model can always be rebuilt exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "GenerationConfig",
    "load_json",
    "save_json",
]


def load_json(path: Union[str, Path]) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class ModelConfig:
    """Architecture of the decoder-only Transformer.

    Sizes are chosen so that a ~1-12M parameter model trains on a 4-core CPU
    with NumPy; the same code scales to larger widths on better hardware.
    """

    # -- size -----------------------------------------------------------
    vocab_size: int = 4096
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 192
    d_ff: Optional[int] = None           # defaults to 4 * d_model
    context_length: int = 256

    # -- structure ------------------------------------------------------
    dropout: float = 0.0
    norm: str = "layernorm"              # "layernorm" | "rmsnorm"
    norm_eps: float = 1e-5
    activation: str = "gelu"             # "gelu" | "relu" | "silu"
    bias: bool = True
    tied_embeddings: bool = True
    pos_embeddings: str = "learned"      # "learned" | "rope"
    rope_theta: float = 10000.0
    init_std: float = 0.02
    residual_init_scale: float = 1.0     # scale of the initialised residual projections

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        if self.norm not in ("layernorm", "rmsnorm"):
            raise ValueError(f"unknown norm {self.norm!r}")
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.pos_embeddings not in ("learned", "rope"):
            raise ValueError(f"unknown positional embedding {self.pos_embeddings!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def estimated_parameters(self) -> int:
        """Analytic parameter count (used before instantiating the model)."""
        v, d, l, f = self.vocab_size, self.d_model, self.n_layer, int(self.d_ff or 4 * self.d_model)
        emb = v * d if self.tied_embeddings else 2 * v * d
        per_layer = 4 * d * d + (4 * d if self.bias else 0)      # attention projections
        per_layer += 2 * f * d + (f + d if self.bias else 0)     # MLP
        per_layer += (2 * d if self.norm == "layernorm" else d)  # two norms
        total = emb + l * per_layer + (2 * d if self.norm == "layernorm" else d)
        if self.pos_embeddings == "learned":
            total += self.context_length * d
        return total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DataConfig:
    """Where training text comes from and how it is turned into sequences."""

    dataset: str = "tinyshakespeare"     # key in data/sources/*.json, or path to a .txt
    tokenizer_path: Optional[str] = None  # None -> build one from the corpus
    vocab_size: int = 4096
    text_field: str = "text"             # for .jsonl sources
    val_fraction: float = 0.05
    min_chars: int = 0
    max_chars: Optional[int] = None
    dedupe: bool = True
    lowercase: bool = False
    cache_dir: str = "data/processed"
    #: SFT evaluation rows are excluded from the pretraining corpus so that a
    #: held-out instruction score is a real generalisation measurement.  Set to
    #: False only for ablation runs; it makes those numbers meaningless.
    exclude_sft_holdout: bool = True
    sft_holdout: int = 16
    #: Instruction file whose evaluation rows must be withheld from pretraining.
    sft_path: str = "data/sources/boxz_instruct_v2.jsonl"
    seed: int = 1234

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainConfig:
    """Hyper-parameters of a single training run."""

    run_name: str = "boxz-small"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 2000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    schedule: str = "cosine"
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    seed: int = 1337
    dtype: str = "float32"
    device: str = "cpu"
    backend: str = "numpy"               # "numpy" | "rust" (falls back to numpy)
    resume: Optional[str] = None
    #: Warm-start: load *weights only* from this checkpoint and begin a new run
    #: (step 0, fresh AdamW moments, fresh LR schedule).  Use this to continue
    #: pretraining on new data.  ``resume`` is different -- it restores step,
    #: optimizer moments and the data stream to continue an interrupted run.
    init_from: Optional[str] = None
    max_minutes: Optional[float] = None  # wall-clock guard for CI / demos

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        model = ModelConfig.from_dict(d.pop("model", {}) or {})
        data = DataConfig.from_dict(d.pop("data", {}) or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        rest = {k: v for k, v in d.items() if k in known and k not in ("model", "data")}
        return cls(model=model, data=data, **rest)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainConfig":
        return cls.from_dict(load_json(path))

    def save(self, path: Union[str, Path]) -> None:
        save_json(self.to_dict(), path)

    def with_overrides(self, **kwargs: Any) -> "TrainConfig":
        return replace(self, **kwargs)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.95
    repetition_penalty: float = 1.0
    repetition_window: int = 64
    stop_tokens: Tuple[str, ...] = ("<|end|>", "<eos>")
    seed: Optional[int] = None
    # Byte-level hygiene.  A model trained on clean text still assigns some
    # probability to control characters and invalid UTF-8 continuation bytes;
    # masking them keeps sampled output printable without touching the weights.
    forbid_control_tokens: bool = True
    utf8_safe: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stop_tokens"] = list(self.stop_tokens)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        d = dict(d)
        if "stop_tokens" in d:
            d["stop_tokens"] = tuple(d["stop_tokens"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

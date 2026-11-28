"""Checkpoint registry: which checkpoint is safe to use, and why.

Nothing is ever deleted or modified here.  A checkpoint that is incomplete or
known worse is *annotated*, and the loader can be asked to refuse it.  The
reason this exists at all: ``checkpoints/v4-sft/best`` was written by a run
that died at step 100 (val 4.70) and was never finished, but it sits in the
tree looking exactly like a usable checkpoint (same size, same field names,
same ``stage`` string).  Loading it by accident silently degrades quality.

The registry is a JSON file, so the answer to "why is this checkpoint marked
bad?" is a diff, not a memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["CheckpointStatus", "load_registry", "checkpoint_status",
           "usable_checkpoints", "require_usable"]

DEFAULT_REGISTRY = Path("checkpoints/registry.json")


class CheckpointStatus:
    """A checkpoint path plus what is known about it."""

    def __init__(self, path: str, data: Dict[str, Any]) -> None:
        self.path = path
        self.data = data

    @property
    def usable(self) -> bool:
        return bool(self.data.get("usable", False))

    @property
    def reason(self) -> str:
        return str(self.data.get("reason", ""))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CheckpointStatus({self.path!r}, usable={self.usable})"

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, **self.data}


def load_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the registry; an absent file is an empty registry, not an error."""
    p = Path(path) if path is not None else DEFAULT_REGISTRY
    if not p.exists():
        return {"format": "boxz-checkpoint-registry-v1", "checkpoints": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def checkpoint_status(path: str, registry_path: Optional[Path] = None) -> CheckpointStatus:
    """Status of ``path``.  Unregistered checkpoints are usable by default.

    Defaulting to usable matters: a freshly trained run must not need a manual
    registry edit before it can be evaluated.
    """
    reg = load_registry(registry_path)
    return CheckpointStatus(path, reg.get("checkpoints", {}).get(path, {"usable": True}))


def usable_checkpoints(registry_path: Optional[Path] = None) -> List[str]:
    reg = load_registry(registry_path)
    return [p for p, d in reg.get("checkpoints", {}).items() if d.get("usable")]


def require_usable(path: str, registry_path: Optional[Path] = None) -> None:
    """Raise if ``path`` is registered as unusable.

    Call this from anything that *selects* a checkpoint for evaluation or
    deployment -- not from forensic tools that deliberately inspect bad ones.
    """
    status = checkpoint_status(path, registry_path)
    if not status.usable:
        raise ValueError(
            f"checkpoint {path!r} is registered as unusable: {status.reason}. "
            f"Use a different checkpoint, or pass a registry that allows it."
        )

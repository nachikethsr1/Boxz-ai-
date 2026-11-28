"""Dataset registry and fetching.

Boxz never silently downloads data.  Every dataset must be declared in
``data/sources/<name>.json`` with its licence, source and checksum, and fetching
is an explicit ``python -m boxz.data.fetch <name>`` step.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["DatasetSource", "list_sources", "get_source", "fetch_source", "datasets_dir"]

_SOURCES_REL = Path("data/sources")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def datasets_dir() -> Path:
    return _repo_root() / "data" / "raw"


@dataclass
class DatasetSource:
    name: str
    title: str
    licence: str
    url: str
    filename: str
    fmt: str                     # "text" | "jsonl"
    text_field: str = "text"
    sha256: Optional[str] = None
    notes: str = ""
    allowed_use: str = ""
    extract_member: Optional[str] = None   # path inside a tar archive, if any

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "DatasetSource":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(name=name, **{k: v for k, v in d.items() if k in known})

    def path(self) -> Path:
        return datasets_dir() / self.filename

    def is_present(self) -> bool:
        return self.path().exists()


def list_sources() -> List[DatasetSource]:
    out: List[DatasetSource] = []
    base = _repo_root() / _SOURCES_REL
    if not base.exists():
        return out
    for p in sorted(base.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        out.append(DatasetSource.from_dict(p.stem, d))
    return out


def get_source(name: str) -> DatasetSource:
    for s in list_sources():
        if s.name == name:
            return s
    raise KeyError(f"unknown dataset {name!r}; known: {[s.name for s in list_sources()]}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_source(name: str, force: bool = False, verbose: bool = True) -> Path:
    """Download a declared dataset into ``data/raw`` (licence-checked by policy).

    Datasets with an empty ``url`` are project-authored files that are committed
    to the repository; for those a missing file is a hard error.
    """
    src = get_source(name)
    dest = src.path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not src.url:
        if not dest.exists():
            raise FileNotFoundError(
                f"dataset {name!r} is project-authored and has no download URL; expected {dest}"
            )
        return dest
    if dest.exists() and not force:
        if verbose:
            print(f"[data] {name}: already present at {dest}")
        return dest
    if verbose:
        print(f"[data] {name}: downloading {src.url}\n       licence: {src.licence}")
    # Use codeload-friendly URL if it is a GitHub blob link.
    url = src.url
    if url.startswith("https://github.com/") and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r:
        payload = r.read()
    if src.extract_member:
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            payload = tar.extractfile(src.extract_member).read()
    tmp.write_bytes(payload)
    tmp.replace(dest)
    if src.sha256:
        got = _sha256(dest)
        if got != src.sha256 and verbose:
            print(f"[data] warning: sha256 mismatch for {name} (expected {src.sha256[:12]}..., got {got[:12]}...)")
    if verbose:
        print(f"[data] {name}: -> {dest} ({dest.stat().st_size} bytes)")
    return dest

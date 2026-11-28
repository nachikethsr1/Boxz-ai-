"""Text cleaning and loading."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

__all__ = ["load_text", "clean_text", "normalise_whitespace"]

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def load_text(path: Path, encoding: str = "utf-8") -> str:
    """Read a text file, tolerating invalid UTF-8 bytes."""
    raw = path.read_bytes()
    text = raw.decode(encoding, errors="replace")
    if "\ufffd" in text:
        # Fall back to latin-1 for files that are plainly not UTF-8.
        alt = raw.decode("latin-1")
        if alt.count("\ufffd") < text.count("\ufffd"):
            text = alt
    return text


def normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text


def clean_text(
    text: str,
    min_chars: int = 0,
    max_chars: Optional[int] = None,
    dedupe: bool = True,
    lowercase: bool = False,
    normalise_unicode: bool = True,
) -> str:
    """Normalise a corpus deterministically.

    Steps: unicode NFC, control-character removal, whitespace normalisation,
    optional lowercasing, optional line-level de-duplication (keeps first
    occurrence, which is adequate for prose and instruction data), truncation.
    """
    if normalise_unicode:
        text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub("", text)
    text = normalise_whitespace(text)
    if lowercase:
        text = text.lower()
    if dedupe:
        seen: set[str] = set()
        out: list[str] = []
        for line in text.split("\n"):
            key = line.strip()
            if key:
                if key in seen:
                    continue
                seen.add(key)
            out.append(line)
        text = "\n".join(out)
    text = text.strip()
    if len(text) < min_chars:
        raise ValueError(f"corpus has {len(text)} chars, below min_chars={min_chars}")
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text

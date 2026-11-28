"""Parse ``coding_requests.txt`` into machine-readable routing records.

The file is the single source of truth for *which requests map to which
template*.  It is plain text on purpose: one record per line, four line types.

    # comment
    === CATEGORY: 2d_arcade_games
    === TEMPLATE: js_snake
    === CLARIFY: game_type_unknown
    === NO_TEMPLATE: language_not_supported
    - Make a Snake game where the tail grows...

Everything after a ``=== TEMPLATE:`` / ``=== CLARIFY:`` / ``=== NO_TEMPLATE:``
header belongs to that block until the next header.  ``- `` lines are the
user requests.  Anything else is a hard error: a silently ignored line is a
silently missing routing example, which is exactly the kind of bug that makes
a router mysteriously fail on one phrasing.

This module only *reads*.  It never writes and never touches the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = [
    "RequestBlock",
    "RequestCorpus",
    "parse_coding_requests",
    "DEFAULT_REQUEST_FILE",
]

DEFAULT_REQUEST_FILE = Path("coding_requests.txt")

_CATEGORY = re.compile(r"^===\s*CATEGORY:\s*(\S+)\s*$")
_BLOCK = re.compile(r"^===\s*(TEMPLATE|CLARIFY|NO_TEMPLATE):\s*(\S+)\s*$")
_REQUEST = re.compile(r"^-\s+(\S.*)$")

#: Block kind -> the machine "intent" it implies.
_KIND_INTENT = {
    "TEMPLATE": "coding",
    "CLARIFY": "clarify",
    "NO_TEMPLATE": "unsupported",
}


@dataclass
class RequestBlock:
    """One ``=== ...: id`` header plus the requests listed under it."""

    kind: str                      # "TEMPLATE" | "CLARIFY" | "NO_TEMPLATE"
    id: str                        # template id, or clarify/unsupported reason
    category: str
    requests: List[str] = field(default_factory=list)
    line: int = 0                  # line number of the header

    @property
    def intent(self) -> str:
        return _KIND_INTENT[self.kind]

    @property
    def template_id(self) -> Optional[str]:
        return self.id if self.kind == "TEMPLATE" else None

    @property
    def reason(self) -> Optional[str]:
        return self.id if self.kind != "TEMPLATE" else None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "id": self.id,
            "category": self.category,
            "intent": self.intent,
            "template_id": self.template_id,
            "reason": self.reason,
            "requests": list(self.requests),
        }


@dataclass
class RequestCorpus:
    """The parsed file."""

    blocks: List[RequestBlock] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    path: Optional[Path] = None

    # -- views -----------------------------------------------------------
    def by_kind(self, kind: str) -> List[RequestBlock]:
        return [b for b in self.blocks if b.kind == kind]

    @property
    def templates(self) -> List[RequestBlock]:
        return self.by_kind("TEMPLATE")

    @property
    def clarifications(self) -> List[RequestBlock]:
        return self.by_kind("CLARIFY")

    @property
    def unsupported(self) -> List[RequestBlock]:
        return self.by_kind("NO_TEMPLATE")

    def template_ids(self) -> List[str]:
        return [b.id for b in self.templates]

    def request_count(self) -> int:
        return sum(len(b.requests) for b in self.blocks)

    def flat_records(self) -> List[dict]:
        """One dict per request line, ready to be written as JSONL."""
        out: List[dict] = []
        for b in self.blocks:
            for req in b.requests:
                out.append({
                    "utterance": req,
                    "intent": b.intent,
                    "template_id": b.template_id,
                    "reason": b.reason,
                    "category": b.category,
                    "block_kind": b.kind,
                })
        return out

    def summary(self) -> dict:
        return {
            "categories": len(self.categories),
            "templates": len(self.templates),
            "clarify_blocks": len(self.clarifications),
            "unsupported_blocks": len(self.unsupported),
            "requests": self.request_count(),
        }


def parse_coding_requests(
    path: str | Path = DEFAULT_REQUEST_FILE,
    strict: bool = True,
) -> RequestCorpus:
    """Parse the request file.

    ``strict`` raises :class:`ValueError` on a malformed line or a request
    that appears before any block header.  Leave it on: a crash at load time
    is far cheaper than a routing example that quietly never matches.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    corpus = RequestCorpus(path=p)
    category: Optional[str] = None
    current: Optional[RequestBlock] = None
    seen_blocks: Dict[str, int] = {}
    seen_requests: Dict[str, int] = {}

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        m = _CATEGORY.match(stripped)
        if m:
            category = m.group(1)
            if category not in corpus.categories:
                corpus.categories.append(category)
            current = None          # a new category closes the open block
            continue

        m = _BLOCK.match(stripped)
        if m:
            if category is None:
                raise ValueError(
                    f"{p}:{lineno}: block {m.group(2)!r} appears before any "
                    f"=== CATEGORY: header"
                )
            block_id = m.group(2)
            if block_id in seen_blocks:
                raise ValueError(
                    f"{p}:{lineno}: duplicate block id {block_id!r} "
                    f"(first defined on line {seen_blocks[block_id]})"
                )
            seen_blocks[block_id] = lineno
            current = RequestBlock(kind=m.group(1), id=block_id,
                                   category=category, line=lineno)
            corpus.blocks.append(current)
            continue

        m = _REQUEST.match(stripped)
        if m:
            if current is None:
                raise ValueError(
                    f"{p}:{lineno}: request {m.group(1)[:40]!r} appears before "
                    f"any === TEMPLATE:/=== CLARIFY:/=== NO_TEMPLATE: header"
                )
            req = m.group(1).strip()
            key = req.lower()
            if key in seen_requests:
                raise ValueError(
                    f"{p}:{lineno}: duplicate request {req[:50]!r} "
                    f"(first seen on line {seen_requests[key]})"
                )
            seen_requests[key] = lineno
            current.requests.append(req)
            continue

        if strict:
            raise ValueError(
                f"{p}:{lineno}: unrecognised line {stripped[:70]!r}. "
                f"Expected '#', '=== CATEGORY:', '=== TEMPLATE:', "
                f"'=== CLARIFY:', '=== NO_TEMPLATE:' or '- '."
            )

    # Blocks with no requests are dead weight: they route nothing.
    empty = [b.id for b in corpus.blocks if not b.requests]
    if empty and strict:
        raise ValueError(f"{p}: block(s) with no requests: {empty}")

    return corpus

"""Decide which engine (if any) handles a user turn.

Two completely different jobs live here:

``route(question)``
    Pre-generation.  For maths it extracts the expression and gets an **exact**
    answer from :mod:`boxz.math.engine`, so the model only has to phrase a
    result it was handed.  For everything else the model speaks for itself.

``postprocess(reply)``
    Post-generation.  Every fenced block in the model's answer is syntax-checked
    by :mod:`boxz.coding.engine` and the verdict is appended as a short, honest
    line -- or *not* appended when the model wrote no code.

The router is stateless: it never stores the "current topic", so a user can go
maths -> coding -> science -> maths across consecutive turns with no reset.
Conversation history belongs to :class:`boxz.chat.engine.ChatEngine`; tool state
belongs to this module.  Keeping them separate is what prevents a maths request
from being contaminated by the previous coding request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .coding.engine import CodeBlock, validate_reply, verdict
from .math.engine import MathAnswer, solve

__all__ = ["Route", "route", "postprocess", "describe"]

#: Questions that look like maths but are really requests for a program.
_CODE_HINTS = re.compile(
    r"\b(write|create|build|make|code|script|program|function|class|"
    r"javascript|html|css|python|three\.?js|component|app|game|website|page)\b",
    re.IGNORECASE,
)

#: Words that make a maths question a *conceptual* question, not a calculation.
_CONCEPT_HINTS = re.compile(
    r"\b(what is|what are|explain|difference between|why|when|define|meaning of|"
    r"describe|how does|inverse|property|properties|commutative|associative|"
    r"distributive|terminating|repeating)\b",
    re.IGNORECASE,
)


@dataclass
class Route:
    used_math: bool = False
    used_coding: bool = False
    math: Optional[MathAnswer] = None
    blocks: List[CodeBlock] = field(default_factory=list)
    coding_verdict: str = ""

    def to_dict(self) -> dict:
        return {
            "used_math": self.used_math,
            "used_coding": self.used_coding,
            "math": ({"ok": self.math.ok, "value": self.math.value,
                      "kind": self.math.kind, "expression": self.math.expression}
                     if self.math else None),
            "code_blocks": [b.validation.to_dict() for b in self.blocks
                            if b.validation is not None],
            "coding_verdict": self.coding_verdict,
        }


def route(question: str) -> Route:
    """Pre-generation routing.  Only maths triggers a tool here."""
    q = (question or "").strip()
    if not q:
        return Route()
    # A request to *write* something is a coding turn even if it contains maths.
    if _CODE_HINTS.search(q) and re.search(r"```|\bcode\b|\bscript\b|\bfunction\b|"
                                           r"\bjavascript\b|\bhtml\b|\bcss\b", q, re.I):
        return Route()
    answer = solve(q)
    if not answer.ok:
        return Route()
    # A pure conceptual question ("what is a variable in algebra?") must not be
    # answered with a bare number; only accept a confident calculation.
    if _CONCEPT_HINTS.search(q) and answer.kind == "arithmetic" and not re.search(
            r"\d\s*[+\-*/^%]\s*\d|\d+\s*(?:plus|minus|times|divided)", q, re.I):
        return Route()
    return Route(used_math=True, math=answer)


def postprocess(reply: str) -> Route:
    """Post-generation: syntax-check every fenced block in the reply."""
    if "```" not in (reply or ""):
        return Route()
    blocks = validate_reply(reply)
    if not blocks:
        return Route()
    return Route(used_coding=True, blocks=blocks, coding_verdict=verdict(blocks))


def should_trust_result(route_obj: Route) -> bool:
    """The model may phrase a verified number but must never replace it."""
    return bool(route_obj.used_math and route_obj.math and route_obj.math.ok)


def describe() -> str:
    return "\n".join([
        "math engine : exact arithmetic, percentages, fractions, powers/roots,",
        "              averages, ratios, unit conversion, geometry, linear algebra",
        "coding engine: fence extraction, language ID, JS/TS/HTML/CSS/Python/JSON",
        "              syntax validation (never executes code)",
        "routing     : per-message, stateless -- topics can change every turn",
    ])

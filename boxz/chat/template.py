"""The single chat template, shared by training and inference.

Training and generation *must* render a conversation byte-for-byte the same
way, otherwise the model is fed a distribution it never saw and produces
gibberish.  This module is the only place that decides how a conversation
becomes text; :mod:`boxz.data.instructions` (SFT targets) and
:mod:`boxz.chat.format` (inference prompts) both call it.

Format::

    <|user|>write a loop<|assistant|>for i in range(3):<|end|>

Turns are concatenated with no separator and no newline, which is exactly what
the SFT dataset renders.  A system turn is optional and is only emitted when
one is actually supplied: the bundled instruction set contains no system
messages, so by default the prompt starts at ``<|user|>``.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "ROLE_MARKERS",
    "END_MARKER",
    "ASSISTANT_OPEN",
    "STOP_MARKERS",
    "render_turn",
    "render_pieces",
    "render_messages",
    "render_prompt",
    "stop_marker",
    "decode_ids_stopping_at_role",
]

ROLE_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
END_MARKER = "<|end|>"
ASSISTANT_OPEN = ROLE_MARKERS["assistant"]
#: Any of these means "the assistant turn is over"; generation must stop.
STOP_MARKERS = (END_MARKER, *ROLE_MARKERS.values())


def render_turn(role: str, content: str) -> str:
    """Render one message.  Only assistant turns are terminated."""
    if role not in ROLE_MARKERS:
        raise ValueError(f"invalid role {role!r}")
    marker = ROLE_MARKERS[role]
    return f"{marker}{content}{END_MARKER if role == 'assistant' else ''}"


def render_pieces(
    messages: Iterable[Mapping[str, str]],
) -> List[Tuple[str, bool]]:
    """Return ``(text, trainable)`` for each message.

    ``trainable`` is True for assistant turns: those are the tokens the SFT
    loss is computed on.  Everything else is prompt context.
    """
    pieces: List[Tuple[str, bool]] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", "")).strip()
        pieces.append((render_turn(role, content), role == "assistant"))
    return pieces


def render_messages(messages: Iterable[Mapping[str, str]]) -> str:
    """Render a whole conversation (no trailing assistant opener)."""
    return "".join(text for text, _ in render_pieces(messages))


def render_prompt(
    messages: Iterable[Mapping[str, str]],
    system: Optional[str] = None,
) -> str:
    """Render conversation history and open a fresh assistant turn."""
    head = ""
    if system is not None and str(system).strip():
        head = render_turn("system", str(system))
    return head + render_messages(messages) + ASSISTANT_OPEN


def stop_marker(text: str) -> Optional[str]:
    """Return the earliest structural marker present in ``text``, if any."""
    best: Optional[str] = None
    best_at = len(text)
    for marker in STOP_MARKERS:
        at = text.find(marker)
        if 0 <= at < best_at:
            best, best_at = marker, at
    return best


def decode_ids_stopping_at_role(ids, tokenizer) -> str:
    """Decode token ids and truncate at the first role/end marker.

    Streaming can emit part of a marker before sampling stops, so a UI must
    never be shown raw control tokens.
    """
    text = tokenizer.decode([int(i) for i in ids])
    marker = stop_marker(text)
    return text[:text.find(marker)] if marker else text

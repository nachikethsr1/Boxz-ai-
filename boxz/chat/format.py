"""Conversation state for the Boxz chat layer.

The actual *text* format lives in :mod:`boxz.chat.template`, which is shared with
instruction tuning: whatever we train the model on is exactly what we feed it at
inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .template import ASSISTANT_OPEN, render_messages, render_prompt

__all__ = ["Message", "Conversation", "SYSTEM_PROMPT"]

#: The instruction data contains no system turns, so the default chat prompt
#: has none either.  A system prompt may be supplied explicitly, but the model
#: was not trained on one and will be out of distribution if you do.
SYSTEM_PROMPT: Optional[str] = None


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"invalid role {self.role!r}")

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """An ordered list of messages plus the token budget policy."""

    system: Optional[str] = SYSTEM_PROMPT
    messages: List[Message] = field(default_factory=list)

    # -- mutation --------------------------------------------------------
    def add_user(self, content: str) -> "Conversation":
        self.messages.append(Message("user", content))
        return self

    def add_assistant(self, content: str) -> "Conversation":
        self.messages.append(Message("assistant", content))
        return self

    def reset(self) -> None:
        self.messages.clear()

    # -- rendering -------------------------------------------------------
    def _all_messages(self, include_system: bool = True) -> List[dict]:
        msgs: List[dict] = []
        if include_system and self.system and self.system.strip():
            msgs.append({"role": "system", "content": self.system.strip()})
        msgs.extend(m.to_dict() for m in self.messages)
        return msgs

    def render(self, include_system: bool = True) -> str:
        return render_messages(self._all_messages(include_system))

    def render_prompt(self) -> str:
        """Render the conversation *and open a fresh assistant turn."""
        return render_prompt([m.to_dict() for m in self.messages], system=self.system)

    # -- budgeting -------------------------------------------------------
    def truncated(self, tokenizer, max_tokens: int, keep_system: bool = True) -> "Conversation":
        """Return a copy whose rendered prompt fits in ``max_tokens``.

        Oldest turns are dropped first; the system prompt and the most recent
        user message are always kept.
        """
        conv = Conversation(system=self.system if keep_system else None,
                            messages=list(self.messages))
        while len(tokenizer.encode(conv.render_prompt())) > max_tokens and len(conv.messages) > 1:
            # drop the oldest non-final exchange
            for i, m in enumerate(conv.messages):
                if i < len(conv.messages) - 1:
                    del conv.messages[i]
                    break
        return conv

    def to_list(self) -> List[dict]:
        return [m.to_dict() for m in self.messages]

    @classmethod
    def from_list(cls, data: Iterable[dict], system: Optional[str] = SYSTEM_PROMPT) -> "Conversation":
        conv = cls(system=system)
        for d in data:
            role = d.get("role")
            if role == "system":
                conv.system = d.get("content", "")
            elif role in ("user", "assistant"):
                conv.messages.append(Message(role, d.get("content", "")))
        return conv

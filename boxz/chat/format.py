"""Conversation formatting.

Boxz has exactly one on-disk/in-context chat format, and the tokenizer knows
its markers as *special tokens* so that user text can never spoof a role
boundary::

    <|user|>write a loop<|end|>... <|assistant|>for i in range(3):<|end|>

(The user turn's closing marker is optional in prompts; generated user text is
never emitted by the model because it only ever writes assistant turns.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

__all__ = ["Message", "Conversation", "SYSTEM_PROMPT"]


SYSTEM_PROMPT = (
    "You are Boxz, a small language model being trained from scratch. "
    "You are honest about your limits and help with programming and games."
)


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
    def render(self, include_system: bool = True) -> str:
        parts: List[str] = []
        if include_system and self.system:
            parts.append(f"<|system|>{self.system.strip()}")
        for m in self.messages:
            if m.role == "assistant":
                parts.append(f"<|assistant|>{m.content.strip()}<|end|>")
            elif m.role == "user":
                parts.append(f"<|user|>{m.content.strip()}")
            else:
                parts.append(f"<|system|>{m.content.strip()}")
        return "\n".join(parts)

    def render_prompt(self) -> str:
        """Render the conversation *and open a fresh assistant turn."""
        base = self.render()
        return base + "\n<|assistant|>"

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

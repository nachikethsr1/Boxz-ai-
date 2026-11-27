"""The chat engine: turns a Conversation into a generated reply.

It is a thin, explicit layer over the model.  There is no canned response
anywhere: :meth:`ChatEngine.reply` formats the conversation with the model's own
special tokens, runs inference, and returns exactly what the model sampled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..config import GenerationConfig
from ..inference.generator import GenerationResult, generate_ids, generate_with_kv_cache
from ..model.transformer import BoxzLM
from ..tokenizer.bpe import BPETokenizer
from .format import SYSTEM_PROMPT, Conversation, Message

__all__ = ["ChatEngine", "ChatReply"]


@dataclass
class ChatReply:
    text: str
    tokens_generated: int
    seconds: float
    finish_reason: str
    prompt_tokens: int
    prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "seconds": round(self.seconds, 3),
            "tokens_per_second": round(self.tokens_generated / max(self.seconds, 1e-9), 2),
            "finish_reason": self.finish_reason,
        }


class ChatEngine:
    def __init__(
        self,
        model: BoxzLM,
        tokenizer: BPETokenizer,
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_context_tokens: Optional[int] = None,
        build_kv_cache: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.conversation = Conversation(system=system_prompt)
        self.max_context_tokens = max_context_tokens or model.cfg.context_length
        self.use_kv_cache = build_kv_cache

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.conversation.reset()

    def history(self) -> List[dict]:
        return self.conversation.to_list()

    # ------------------------------------------------------------------
    def build_prompt(self) -> str:
        conv = self.conversation.truncated(self.tokenizer, self.max_context_tokens)
        self.conversation = conv
        return conv.render_prompt()

    def reply(self, user_message: str, cfg: Optional[GenerationConfig] = None) -> ChatReply:
        cfg = cfg or GenerationConfig()
        self.conversation.add_user(user_message)
        prompt = self.build_prompt()
        prompt_ids = self.tokenizer.encode(prompt)

        stop_ids = [
            self.tokenizer.special_id("<|end|>"),
            self.tokenizer.special_id("<|user|>"),
            self.tokenizer.special_id("<|system|>"),
            self.tokenizer.special_id("<eos>"),
        ]

        if self.use_kv_cache and hasattr(self.model, "blocks"):
            # The KV-cache path applies the same byte-hygiene mask as generate_ids.
            res = generate_with_kv_cache(self.model, self.tokenizer, prompt, cfg)
        else:
            res = generate_ids(self.model, prompt_ids, cfg, self.tokenizer, stop_ids=stop_ids)

        text = self._clean(res.text)
        self.conversation.add_assistant(text)
        return ChatReply(
            text=text,
            tokens_generated=res.tokens_generated,
            seconds=res.seconds,
            finish_reason=res.finish_reason,
            prompt_tokens=len(prompt_ids),
            prompt=prompt,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        """Strip anything after the first turn terminator and normalise spaces."""
        for marker in ("<|end|>", "<|user|>", "<|system|>", "<|assistant|>"):
            if marker in text:
                text = text.split(marker)[0]
        return text.strip()

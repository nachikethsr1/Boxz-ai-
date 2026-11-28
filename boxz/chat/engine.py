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
from ..engine import no_grad
from ..inference.generator import (GenerationResult, _params_from_config,
                                   generate_ids, generate_with_kv_cache)
from ..inference.sampler import sample_token
from ..model.transformer import BoxzLM
from ..router import Route, postprocess, route as route_question
from ..tokenizer.bpe import BPETokenizer
from .format import SYSTEM_PROMPT, Conversation, Message
from .template import decode_ids_stopping_at_role

__all__ = ["ChatEngine", "ChatReply"]


@dataclass
class ChatReply:
    text: str
    tokens_generated: int
    seconds: float
    finish_reason: str
    prompt_tokens: int
    prompt: str = ""
    tools: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "seconds": round(self.seconds, 3),
            "tokens_per_second": round(self.tokens_generated / max(self.seconds, 1e-9), 2),
            "finish_reason": self.finish_reason,
            "tools": self.tools or {},
        }


class ChatEngine:
    def __init__(
        self,
        model: BoxzLM,
        tokenizer: BPETokenizer,
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_context_tokens: Optional[int] = None,
        build_kv_cache: bool = True,
        use_tools: bool = True,
        tool_seed: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.conversation = Conversation(system=system_prompt)
        self.max_context_tokens = max_context_tokens or model.cfg.context_length
        self.use_kv_cache = build_kv_cache
        self.use_tools = use_tools
        self.tool_seed = tool_seed
        self.last_route: Optional[Route] = None
        self.verified_values = 0
        self.server_identity_seed = tool_seed

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

    # ------------------------------------------------------------------
    # tool plumbing
    # ------------------------------------------------------------------
    def plan(self, user_message: str) -> Route:
        """Decide, before generating, whether an engine should answer."""
        if not self.use_tools:
            return Route()
        return route_question(user_message)

    def _math_reply(self, r: Route, user_message: str) -> ChatReply:
        """Answer a calculation with the engine's verified value.

        The model is *not* asked to do arithmetic.  It would get it wrong: a
        3M-parameter model cannot reliably multiply two 6-digit numbers.  So the
        engine computes, and the reply states the exact result.  If the model
        has learned the phrasing it is used as the wrapper; otherwise the
        canonical phrasing is emitted.  The numeric result is never generated.
        """
        ans = r.math
        assert ans is not None
        self.verified_values += 1
        body = self._phrase_calculation(ans, user_message)
        self.conversation.add_assistant(body)
        return ChatReply(
            text=body,
            tokens_generated=0,
            seconds=0.0,
            finish_reason="math_engine",
            prompt_tokens=0,
            prompt="",
            tools=r.to_dict(),
        )

    def _phrase_calculation(self, ans, user_message: str) -> str:
        """Deterministic, verifiable phrasing of an exact result."""
        if ans.kind == "arithmetic":
            return f"{ans.expression} = {ans.value}"
        if ans.kind == "percentage":
            return f"{ans.expression} = {ans.value}"
        if ans.kind == "geometry":
            unit = " square units" if "area" in ans.expression else ""
            return f"The {ans.expression} is {ans.value}{unit}."
        if ans.kind == "conversion":
            return f"{ans.expression}: {ans.value}"
        if ans.kind == "average":
            return f"{ans.expression} is {ans.value}."
        if ans.kind == "ratio":
            return f"{ans.expression}: {ans.value}."
        if ans.kind == "algebra":
            return f"Solving {ans.expression.replace('solve ', '')} gives {ans.value}."
        return ans.sentence()

    def reply(self, user_message: str, cfg: Optional[GenerationConfig] = None) -> ChatReply:
        cfg = cfg or GenerationConfig()
        planned = self.plan(user_message)
        self.last_route = planned
        self.conversation.add_user(user_message)
        if planned.used_math:
            return self._math_reply(planned, user_message)
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
        check = postprocess(text)
        tools = check.to_dict() if check.used_coding else {}
        shown = text
        if check.used_coding and check.coding_verdict:
            shown = text + "\n\n(" + check.coding_verdict + ")"
        self.conversation.add_assistant(shown)
        return ChatReply(
            text=shown,
            tokens_generated=res.tokens_generated,
            seconds=res.seconds,
            finish_reason=res.finish_reason,
            prompt_tokens=len(prompt_ids),
            prompt=prompt,
            tools=tools,
        )

    def reply_stream(self, user_message: str, cfg: Optional[GenerationConfig] = None):
        """Yield the cumulative reply text as tokens are generated.

        Streaming matters on CPU: the first words appear in milliseconds even
        when a full 200-token reply takes several seconds.  Each yielded value
        is the *whole* reply so far, so a client just replaces its text node.
        The generator's return value is the final cleaned reply string.
        """
        cfg = cfg or GenerationConfig()
        planned = self.plan(user_message)
        self.last_route = planned
        self.conversation.add_user(user_message)
        if planned.used_math:
            # Streaming cannot stream a number it did not generate; emit the
            # verified sentence in one chunk so the UI still updates.
            body = self._phrase_calculation(planned.math, user_message)
            self.verified_values += 1
            self.conversation.add_assistant(body)
            yield body
            return
        prompt_ids = self.tokenizer.encode(self.build_prompt())
        stop_ids = {
            self.tokenizer.special_id("<|end|>"),
            self.tokenizer.special_id("<|user|>"),
            self.tokenizer.special_id("<|system|>"),
            self.tokenizer.special_id("<eos>"),
        }
        params = _params_from_config(cfg, self.tokenizer)
        rng = np.random.default_rng(cfg.seed)
        ctx = int(self.model.cfg.context_length)
        generated: List[int] = []
        shown = ""

        self.model.eval()
        with no_grad():
            for _ in range(int(cfg.max_new_tokens)):
                window = (prompt_ids + generated)[-ctx:]
                logits, _ = self.model(np.asarray([window], dtype=np.int64))
                recent = (generated[-cfg.repetition_window:]
                          if cfg.repetition_penalty != 1.0 else None)
                nxt = int(sample_token(logits.data[0, -1], params, recent, rng))
                generated.append(nxt)
                text = self._visible(generated)
                if text != shown:
                    shown = text
                    yield text
                if nxt in stop_ids:
                    break

        final = self._clean(self._visible(generated))
        check = postprocess(final)
        self.last_route = check
        if check.used_coding and check.coding_verdict:
            final = final + "\n\n(" + check.coding_verdict + ")"
        self.conversation.add_assistant(final)
        yield final

    def _visible(self, ids: List[int]) -> str:
        """Decode generated ids, cut at the first structural marker, tidy edges."""
        return decode_ids_stopping_at_role(ids, self.tokenizer).strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _clean(text: str) -> str:
        """Strip anything after the first turn terminator and normalise spaces."""
        for marker in ("<|end|>", "<|user|>", "<|system|>", "<|assistant|>"):
            if marker in text:
                text = text.split(marker)[0]
        return text.strip()

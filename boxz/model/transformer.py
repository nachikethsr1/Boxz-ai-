"""The Boxz decoder-only Transformer language model.

    token ids -> token embedding (+ positional encoding)
              -> N x [ LayerNorm -> causal MHA -> residual -> LayerNorm -> MLP -> residual ]
              -> final LayerNorm -> output projection -> logits over vocabulary

Every operation is a Boxz ``Function``; nothing here delegates learning to a
third-party framework.  The class also owns weight initialisation (with an
optional residual-projection scaling), a parameter report, and state-dict
conversion that is exactly compatible with checkpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ModelConfig
from ..engine import ops
from ..engine.layers import Dropout, Embedding, LayerNorm, Linear, Module, Parameter, RMSNorm
from ..engine.tensor import Tensor, matmul
from .attention import CausalSelfAttention

__all__ = ["BoxzLM", "TransformerBlock", "build_model"]


def _norm_layer(cfg: ModelConfig, dim: int) -> Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(dim, eps=cfg.norm_eps)
    return LayerNorm(dim, eps=cfg.norm_eps)


def _activate(x: Tensor, kind: str) -> Tensor:
    if kind == "gelu":
        return ops.gelu(x)
    if kind == "relu":
        return ops.relu(x)
    if kind == "silu":
        return ops.silu(x)
    raise ValueError(f"unknown activation {kind!r}")


class MLP(Module):
    """Position-wise feed-forward network (pre-norm transformer)."""

    def __init__(self, cfg: ModelConfig, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        d, f = cfg.d_model, int(cfg.d_ff or 4 * cfg.d_model)
        res_std = cfg.init_std / np.sqrt(max(1, 2 * cfg.n_layer)) * cfg.residual_init_scale
        self.register_module("fc_in", Linear(d, f, bias=cfg.bias, rng=rng, std=cfg.init_std))
        self.register_module("fc_out", Linear(f, d, bias=cfg.bias, rng=rng, std=res_std))
        self.dropout = Dropout(cfg.dropout)
        self.activation = cfg.activation

    def forward(self, x: Tensor) -> Tensor:
        h = _activate(self.fc_in(x), self.activation)
        h = self.fc_out(h)
        return self.dropout(h)


class TransformerBlock(Module):
    """Pre-norm block: x + Attn(Norm(x)); x + MLP(Norm(x))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        d = cfg.d_model
        res_std = cfg.init_std / np.sqrt(max(1, 2 * cfg.n_layer)) * cfg.residual_init_scale
        self.layer_idx = layer_idx
        self.register_module("ln_attn", _norm_layer(cfg, d))
        self.register_module("attn", CausalSelfAttention(
            d_model=d, n_head=cfg.n_head, context_length=cfg.context_length,
            bias=cfg.bias, dropout=cfg.dropout, rng=rng, residual_std=res_std,
        ))
        self.register_module("ln_mlp", _norm_layer(cfg, d))
        self.register_module("mlp", MLP(cfg, rng=rng))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x


class BoxzLM(Module):
    def __init__(self, cfg: ModelConfig, seed: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        d, v = cfg.d_model, cfg.vocab_size

        self.register_module("tok_emb", Embedding(v, d, rng=rng, std=cfg.init_std))
        if cfg.pos_embeddings == "learned":
            pos = rng.normal(0.0, cfg.init_std, size=(cfg.context_length, d)).astype(np.float32)
            self.register_parameter("pos_emb", Parameter(pos))
        else:
            self.register_parameter("pos_emb", None)  # type: ignore[arg-type]

        self.blocks: List[TransformerBlock] = []
        for i in range(cfg.n_layer):
            blk = TransformerBlock(cfg, i, rng=rng)
            self.blocks.append(blk)
            self.register_module(f"blocks.{i}", blk)

        self.register_module("ln_final", _norm_layer(cfg, d))
        self.dropout = Dropout(cfg.dropout)
        if not cfg.tied_embeddings:
            lm_head = Linear(d, v, bias=False, rng=rng, std=cfg.init_std)
            self.register_module("lm_head", lm_head)
        else:
            self.register_module("lm_head", None)  # type: ignore[arg-type]
        self._rng = rng

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, tokens: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the model on integer token ids ``(B, T)``.

        Returns ``(logits, loss)``; ``logits`` has shape ``(B, T, vocab)`` and
        ``loss`` is the mean next-token cross entropy when targets are given.
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (batch, seq); got shape {tokens.shape}")
        B, T = tokens.shape
        if T > self.cfg.context_length:
            raise ValueError(f"sequence length {T} > context_length {self.cfg.context_length}")

        x = self.tok_emb(tokens)                                # (B, T, D)
        if self.pos_emb is not None:
            pos = self.pos_emb[...][:T]                          # (T, D)
            if pos.shape != (T, self.cfg.d_model):
                raise ValueError("positional embedding table shorter than sequence")
            x = x + pos
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = matmul(x.reshape(B * T, self.cfg.d_model), self.tok_emb.weight.transpose())
            logits = logits.reshape(B, T, self.cfg.vocab_size)
        loss = None
        if targets is not None:
            loss = ops.cross_entropy_with_logits(logits, np.asarray(targets, dtype=np.int64))
        return logits, loss

    # ------------------------------------------------------------------
    # introspection / persistence
    # ------------------------------------------------------------------
    def parameter_report(self) -> List[Tuple[str, int]]:
        return [(n, int(p.size)) for n, p in self.named_parameters()]

    def summary(self) -> str:
        total = self.num_parameters()
        lines = [f"BoxzLM  d_model={self.cfg.d_model} n_layer={self.cfg.n_layer} "
                 f"n_head={self.cfg.n_head} ctx={self.cfg.context_length} vocab={self.cfg.vocab_size}",
                 f"  parameters: {total:,} ({total * 4 / 1e6:.2f} MB fp32)"]
        return "\n".join(lines)


def build_model(cfg: ModelConfig, seed: Optional[int] = None) -> BoxzLM:
    return BoxzLM(cfg, seed=seed)

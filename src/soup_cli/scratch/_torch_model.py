"""Torch implementation of the from-scratch transformer (lazy import only).

Kept in its own module so importing :mod:`soup_cli.scratch.model` never
pulls in torch at module top level.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — standard torch idiom

from soup_cli.scratch.model import ScratchModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class _Norm(nn.Module):
    def __init__(self, cfg: ScratchModelConfig):
        super().__init__()
        if cfg.norm_type == "rms":
            self.norm: nn.Module = RMSNorm(cfg.d_model, cfg.norm_eps)
        else:
            self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, cfg: ScratchModelConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.ffn_multiplier
        if cfg.activation == "swiglu":
            # gate: half the hidden, value: half the hidden
            self.gate_proj = nn.Linear(cfg.d_model, hidden // 2, bias=False)
            self.value_proj = nn.Linear(cfg.d_model, hidden // 2, bias=False)
            self.out_proj = nn.Linear(hidden // 2, cfg.d_model, bias=False)
        else:
            self.fc_in = nn.Linear(cfg.d_model, hidden, bias=False)
            self.fc_out = nn.Linear(hidden, cfg.d_model, bias=False)
        self.activation = cfg.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))
        if self.activation == "relu":
            return self.fc_out(F.relu(self.fc_in(x)))
        return self.fc_out(F.gelu(self.fc_in(x)))


class CausalAttention(nn.Module):
    def __init__(self, cfg: ScratchModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(b, t, c)
        return self.proj(attn)


class Block(nn.Module):
    def __init__(self, cfg: ScratchModelConfig):
        super().__init__()
        self.norm1 = _Norm(cfg)
        self.attn = CausalAttention(cfg)
        self.norm2 = _Norm(cfg)
        self.ffn = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ScratchTorchModule(nn.Module):
    """The from-scratch causal LM. All weights start at random seeded init."""

    def __init__(self, cfg: ScratchModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.positional_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = _Norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Any:
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.token_embedding(idx) + self.positional_embedding(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return loss, logits
        return self.lm_head(x), None

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling for inference."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, -1, None]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

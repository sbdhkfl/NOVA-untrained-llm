"""A small causal transformer built from scratch — no pretrained weights.

Implements the full architecture stack required by the SOUP training
programme: token embedding, learned positional embeddings, causal multi-head
attention, feed-forward layers, pre-norm RMSNorm, residual connections, and a
language-model output head. All parameters are initialised with a seeded
initialisation scheme so the model genuinely begins without pretrained
knowledge and is reproducible.

Torch is imported lazily inside functions (heavy deps are never at module
top level, per repo convention).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

from rich.console import Console

console = Console()


@dataclass
class ScratchModelConfig:
    """Configuration for a from-scratch transformer. Single source of truth
    for the architecture; self-contained (no ``soup.yaml`` schema changes),
    serialised into dataset plans, remote payloads and checkpoints."""

    vocab_size: int = 256
    block_size: int = 256  # context length / positional embedding size
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    ffn_multiplier: int = 4  # hidden dim = d_model * ffn_multiplier
    dropout: float = 0.0
    norm_eps: float = 1e-5
    norm_type: str = "rms"  # "rms" | "layernorm"
    activation: str = "gelu"  # "gelu" | "relu" | "swiglu"
    tie_weights: bool = True  # tie output head to token embedding
    seed: int = 1337
    architecture: str = "causal-transformer"

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if self.d_model < self.n_heads:
            raise ValueError("d_model must be >= n_heads")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ffn_multiplier < 1:
            raise ValueError("ffn_multiplier must be >= 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_type not in ("rms", "layernorm"):
            raise ValueError("norm_type must be 'rms' or 'layernorm'")
        if self.activation not in ("gelu", "relu", "swiglu"):
            raise ValueError("activation must be 'gelu', 'relu' or 'swiglu'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScratchModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def param_count(self) -> int:
        """Static parameter count estimate (torch-free, for display)."""
        d = self.d_model
        h = self.ffn_multiplier * d
        # per layer: 4 attn matrices (d*d) + 2 ffn (d*h) + 2 norms (2*d)
        per_layer = 4 * d * d + 2 * d * h + 2 * d
        n_embed = self.vocab_size * d
        pos = self.block_size * d
        head = d if self.tie_weights else self.vocab_size * d
        return n_embed + pos + self.n_layers * per_layer + head + d


class ScratchTransformer:
    """A causal transformer built from scratch. All parameters start at
    seeded random init — there is no pretrained checkpoint anywhere in this
    code path. The class is a thin typed shell over the lazy torch module
    so that the rest of the package can be imported without torch."""

    def __init__(self, config: ScratchModelConfig):
        self.config = config
        self._torch_module: Optional[Any] = None

    def build(self) -> Any:
        """Materialise the torch module (requires torch)."""
        import torch as _torch

        from soup_cli.scratch._torch_model import ScratchTorchModule

        module = ScratchTorchModule(self.config)
        _torch.manual_seed(self.config.seed)
        module.apply(ScratchTorchModule._init_weights)
        self._torch_module = module
        return module

    @property
    def torch_module(self) -> Any:
        if self._torch_module is None:
            return self.build()
        return self._torch_module

    def num_parameters(self) -> int:
        """Actual (non-embedding) parameter count from the live module."""
        if self._torch_module is not None:
            return sum(p.numel() for p in self._torch_module.parameters())
        return self.config.param_count()

    def save_config(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            payload = {"architecture": self.config.architecture, **self.config.to_dict()}
            json.dump(payload, fh, indent=2)


def causal_mask(seq_len: int, dtype: Any, device: Any) -> Any:
    """Boolean causal mask; True where attention is blocked."""
    import torch

    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    return ~mask  # torch sdpa wants True = disallowed


def _init_weights(module: Any) -> None:
    """Seeded random init: GPT-2 style — normal(0, 0.02) for linear and embed
    layers, zero for biases. This is the ONLY initialisation path; the model
    genuinely starts without pretrained knowledge."""
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

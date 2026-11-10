"""Checkpoint tensor I/O for the from-scratch model.

Prefers ``safetensors`` (safe, no pickle) and falls back to a torch
``state_dict`` pickle when safetensors is unavailable. Kept in a separate
module so importing :mod:`soup_cli.scratch.engine` never pulls torch at
module top level.
"""

from __future__ import annotations

import os
from typing import Any


def save_model_state(model: Any, path: str) -> None:
    """Save model tensors to ``path`` (safetensors when available)."""
    import torch

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Tied weights share storage (e.g. lm_head.weight <-> token_embedding.weight);
    # safetensors refuses shared tensors, so keep one tensor per storage.
    seen: set[int] = set()
    deduped: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        deduped[key] = tensor
    state = deduped
    try:
        from safetensors.torch import save_file

        save_file(state, path)
    except ImportError:
        torch.save(state, path)


def load_model_state(model: Any, path: str) -> None:
    """Load model tensors from ``path`` into ``model``."""
    import torch

    if os.path.exists(path) and path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file

            state = load_file(path)
        except ImportError:
            # fall back to torch pickle
            state = torch.load(path, weights_only=True)
    else:
        state = torch.load(path, weights_only=True)
    # strict=False: tied weights that were deduplicated on save are restored
    # implicitly (they share storage with the surviving tensor).
    model.load_state_dict(state, strict=False)

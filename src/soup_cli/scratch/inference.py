"""Inference for a trained from-scratch checkpoint.

Loads a downloaded remote-run checkpoint (``checkpoint-final``) plus its
tokenizer and generates text from a prompt. This is *inference only* — a
forward pass over an existing model; nothing here trains or updates
weights, and no pretrained weights are involved anywhere in this path.
"""

from __future__ import annotations

import os
from typing import Any

from soup_cli.scratch.learn import resolve_checkpoint_dir


def load_run(run_dir: str) -> tuple[Any, Any, Any]:
    """Load ``(model, tokenizer, model_config)`` from a run directory.

    ``run_dir`` may be the run directory (containing ``checkpoint-final/``)
    or the checkpoint directory itself. Raises ``FileNotFoundError`` when no
    downloaded checkpoint is present.
    """
    from soup_cli.scratch.engine import CheckpointManager
    from soup_cli.scratch.model import ScratchModelConfig, ScratchTransformer
    from soup_cli.scratch.tokenizer import ScratchTokenizer

    ckpt_dir = resolve_checkpoint_dir(run_dir)
    checkpoint = CheckpointManager(ckpt_dir)
    state = checkpoint.load()
    config = ScratchModelConfig.from_dict(state["model_config"])
    model = ScratchTransformer(config).build()
    checkpoint.restore_model(model)
    model.eval()
    tokenizer = ScratchTokenizer.load(os.path.join(ckpt_dir, "tokenizer.json"))
    return model, tokenizer, config


def generate_from_run(
    run_dir: str,
    prompt: str,
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 0,
) -> str:
    """Generate a continuation for ``prompt`` from a downloaded checkpoint.

    ``temperature=0`` is greedy (deterministic); ``> 0`` samples with a
    seeded generator so runs are reproducible.
    """
    import torch

    model, tokenizer, config = load_run(run_dir)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)[:-1]  # drop eos
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]
    torch.manual_seed(seed)
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    generated = out[0, len(prompt_ids) :].tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)

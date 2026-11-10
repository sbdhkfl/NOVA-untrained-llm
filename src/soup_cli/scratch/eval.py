"""Evaluation for the from-scratch model.

Computes validation loss, perplexity and tokenisation/dataset statistics,
and writes an evaluation report. Nothing here is fabricated: all numbers
come from a real forward pass over real tokenised data.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from rich.console import Console

console = Console()


def evaluate_model(
    *,
    model: Any,
    val_ids: Any,
    block_size: int,
    batch_size: int = 8,
    eval_batches: int = 50,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run real evaluation. Returns loss / perplexity statistics computed
    from actual forward passes."""
    import torch

    if len(val_ids) < block_size + 1:
        raise ValueError(
            f"validation stream too short ({len(val_ids)} tokens) for block_size {block_size}"
        )
    model.to(device)
    model.eval()
    generator = torch.Generator().manual_seed(0)
    losses = []
    n_batches = max(eval_batches, 1)
    with torch.no_grad():
        for _ in range(n_batches):
            ix = torch.randint(len(val_ids) - block_size - 1, (batch_size,), generator=generator)
            x = torch.stack(
                [torch.from_numpy(val_ids[i : i + block_size].astype("int64")) for i in ix]
            ).to(device)
            y = torch.stack(
                [torch.from_numpy(val_ids[i + 1 : i + block_size + 1].astype("int64")) for i in ix]
            ).to(device)
            loss, _ = model(x, y)
            losses.append(float(loss.item()))
    mean_loss = sum(losses) / len(losses)
    # Perplexity from the cross-entropy loss in natural log space.
    perplexity = math.exp(min(mean_loss, 20))
    return {
        "val_loss": mean_loss,
        "val_perplexity": perplexity,
        "eval_batches": n_batches,
        "batch_size": batch_size,
        "block_size": block_size,
        "tokens_per_batch": batch_size * block_size,
    }


def dataset_statistics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Dataset-level statistics for the evaluation report."""
    split = manifest.get("split", {})
    tok = manifest.get("tokenizer", {})
    return {
        "dataset_version": manifest.get("version"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "train_tokens": split.get("train_tokens"),
        "val_tokens": split.get("val_tokens"),
        "train_examples": split.get("train_examples"),
        "val_examples": split.get("val_examples"),
        "vocab_size": tok.get("vocab_size"),
        "tokenizer_hash": tok.get("version_hash"),
    }


def write_report(report: dict[str, Any], path: str) -> str:
    """Write an evaluation report as JSON. Returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    console.print(f"[green]evaluation report written:[/green] {path}")
    return path


def load_report(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

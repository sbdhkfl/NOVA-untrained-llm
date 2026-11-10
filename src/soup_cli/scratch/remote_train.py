"""The remote training worker — runs INSIDE the Modal container only.

This module is the entry point the Modal stub invokes. It never runs on the
user's computer: the local CLI only ever renders plans and submits stubs.
Everything the worker needs arrives in the base64 payload.

Failure modes are reported loudly. There is no silent fallback to local
training anywhere in this path.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

from rich.console import Console

console = Console()


def run_remote_training(payload: dict[str, Any], *, output_dir: str) -> dict[str, Any]:
    """Train the from-scratch model remotely. Called only from the Modal
    container. Returns the run result dict and writes checkpoints to
    ``output_dir``."""
    if payload.get("format") != "soup-scratch-remote":
        raise ValueError("unexpected remote payload format")

    from soup_cli.scratch.engine import ScratchTrainer, ScratchTrainingConfig
    from soup_cli.scratch.model import ScratchModelConfig, ScratchTransformer

    model_cfg = ScratchModelConfig.from_dict(payload["model_config"])
    train_cfg = ScratchTrainingConfig.from_dict(payload["training_config"])

    # Remote runs always target the GPU allocated by Modal.
    train_cfg.device = "cuda"
    manifest = payload["dataset_manifest"]
    ds_version = manifest.get("version", "remote")

    # Materialise the dataset into the container workspace so the trainer
    # can read the token streams the same way as local tooling does.
    work_dir = "/root/scratch-work"
    ds_dir = os.path.join(work_dir, "datasets", ds_version)
    os.makedirs(ds_dir, exist_ok=True)
    import numpy as np

    train_ids = np.load(io.BytesIO(base64.b64decode(payload["train_ids_b64"])))
    val_ids = np.load(io.BytesIO(base64.b64decode(payload["val_ids_b64"])))

    tokenizer_path = os.path.join(ds_dir, "tokenizer.json")
    with open(tokenizer_path, "w", encoding="utf-8") as fh:
        fh.write(payload["tokenizer_json"])

    # Build the model from scratch — seeded random init, no pretrained
    # weights anywhere in this path.
    model = ScratchTransformer(model_cfg).build()

    trainer = ScratchTrainer(
        model=model,
        train_ids=train_ids,
        val_ids=val_ids,
        config=train_cfg,
        checkpoint_dir=os.path.join(output_dir, "checkpoint-final"),
        tokenizer_path=tokenizer_path,
        model_config=model_cfg.to_dict(),
    )
    result = trainer.train(resume=False)

    # Write the run-level metadata for the registry.
    report = {
        "steps": result.steps,
        "epochs": result.epochs,
        "final_train_loss": result.final_train_loss,
        "final_val_loss": result.final_val_loss,
        "final_val_perplexity": result.final_val_perplexity,
        "total_time_sec": result.total_time_sec,
        "checkpoint_dir": result.checkpoint_dir,
        "model_config": model_cfg.to_dict(),
        "training_config": train_cfg.to_dict(),
        "dataset_manifest": manifest,
    }
    with open(os.path.join(output_dir, "run_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    console.print("[bold green]remote from-scratch training complete[/bold green]")
    return report

"""``soup learn`` — the from-scratch learning workflow.

Ingest learning material, review and approve it, build a versioned tokenised
dataset, plan (and optionally submit) REMOTE training on Modal, evaluate the
resulting checkpoint, and register the trained model. Actual training always
runs remotely; the local machine only renders plans and evaluates downloaded
checkpoints.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    help=(
        "Learn a language model from scratch: ingest material, review/approve "
        "examples, then train REMOTELY (Modal) and register the result."
    ),
    no_args_is_help=True,
)


def _workflow(root: str):
    from soup_cli.scratch.learn import LearnWorkflow

    return LearnWorkflow(root)


@app.command("ingest")
def ingest(
    file: Optional[str] = typer.Argument(None, help="Text file to ingest."),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Inline text to ingest."),
    source: Optional[str] = typer.Option(None, "--source", help="Provenance label."),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
) -> None:
    """Ingest learning material (lands in pending — NOT approved yet)."""
    wf = _workflow(root)
    if bool(file) == bool(text):
        raise typer.BadParameter("give exactly one of FILE or --text")
    if file:
        from soup_cli.utils.paths import is_under_cwd

        if not is_under_cwd(file):
            raise typer.BadParameter(f"file must stay under the current working directory: {file}")
        examples = wf.ingest_file(file, source=source)
    else:
        assert text is not None
        examples = [wf.ingest_text(text, source=source or "inline")]
    accepted = sum(1 for e in examples if e["status"] == "pending")
    rejected = sum(1 for e in examples if e["status"] == "rejected")
    console.print(f"Ingested {len(examples)} examples: {accepted} pending, {rejected} rejected.")
    console.print("Review with [bold]soup learn review[/bold] then approve.")


@app.command("review")
def review(
    root: str = typer.Option("learn", "--root"),
    show: int = typer.Option(10, "--show", "-n", help="Examples to preview per bucket."),
) -> None:
    """Review pending / approved / rejected examples."""
    state = _workflow(root).review()
    for bucket in ("pending", "approved", "rejected"):
        table = Table(title=bucket)
        table.add_column("id")
        table.add_column("chars")
        table.add_column("source")
        table.add_column("preview")
        for e in state[bucket][:show]:
            table.add_row(e["example_id"], str(len(e["text"])), e["source"], e["text"][:60])
        console.print(table)
        if len(state[bucket]) > show:
            console.print(f"… and {len(state[bucket]) - show} more")


@app.command("approve")
def approve(
    example_ids: Optional[list[str]] = typer.Argument(None),
    all_pending: bool = typer.Option(False, "--all", help="Approve every pending example."),
    root: str = typer.Option("learn", "--root"),
) -> None:
    """Explicitly approve examples for training (never automatic)."""
    wf = _workflow(root)
    if not all_pending and not example_ids:
        raise typer.BadParameter("give example ids or --all")
    approved = wf.approve(example_ids, all_pending=all_pending)
    console.print(f"[green]Approved {len(approved)} examples.[/green]")


@app.command("reject")
def reject(
    example_ids: list[str] = typer.Argument(...),
    root: str = typer.Option("learn", "--root"),
) -> None:
    """Reject pending examples (they will never be trained on)."""
    rejected = _workflow(root).reject(list(example_ids))
    console.print(f"Rejected {len(rejected)} examples.")


@app.command("prepare")
def prepare(
    version: str = typer.Option(..., "--version", "-v", help="Dataset version tag."),
    vocab_size: int = typer.Option(512, "--vocab-size", min=260),
    val_fraction: float = typer.Option(0.1, "--val-fraction", min=0.01, max=0.5),
    seed: int = typer.Option(1337, "--seed"),
    root: str = typer.Option("learn", "--root"),
) -> None:
    """Tokenise APPROVED examples into a versioned train/val dataset."""
    manifest = _workflow(root).prepare_dataset(
        version=version, vocab_size=vocab_size, val_fraction=val_fraction, seed=seed
    )
    console.print_json(json.dumps(manifest["split"]))


@app.command("train")
def train(
    dataset_version: str = typer.Option(..., "--dataset", "-d"),
    gpu: str = typer.Option("a10g", "--gpu", help="Modal GPU type."),
    submit: bool = typer.Option(False, "--cloud-submit", help="Actually submit the remote run."),
    epochs: int = typer.Option(3, "--epochs", min=1),
    batch_size: int = typer.Option(8, "--batch-size", min=1),
    learning_rate: float = typer.Option(3e-4, "--lr", min=1e-8),
    n_layers: int = typer.Option(4, "--layers", min=1),
    n_heads: int = typer.Option(4, "--heads", min=1),
    d_model: int = typer.Option(128, "--d-model", min=8),
    block_size: int = typer.Option(256, "--block-size", min=16),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    root: str = typer.Option("learn", "--root"),
) -> None:
    """Plan (and optionally submit) REMOTE from-scratch training on Modal.

    This command NEVER trains locally. Without --cloud-submit it is
    plan-only: it renders the Modal stub and prints the planned command.
    """
    from soup_cli.scratch.engine import ScratchTrainingConfig
    from soup_cli.scratch.model import ScratchModelConfig

    wf = _workflow(root)
    model_config = ScratchModelConfig(
        n_layers=n_layers, n_heads=n_heads, d_model=d_model, block_size=block_size
    ).to_dict()
    training_config = ScratchTrainingConfig(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, block_size=block_size
    ).to_dict()
    plan = wf.plan_remote_training(
        dataset_version=dataset_version,
        model_config=model_config,
        training_config=training_config,
        gpu=gpu,
        run_name=run_name,
    )
    console.print(Panel(plan.stub_text[:800] + "\n…", title="Modal stub (plan-only)"))
    console.print(f"Planned run: [bold]{plan.run_command}[/bold]")
    console.print(f"GPU: {plan.gpu}  output: {plan.output_dir}")
    if submit:
        from soup_cli.scratch.remote import submit_scratch_modal_run

        code = submit_scratch_modal_run(plan)
        if code != 0:
            raise typer.Exit(code=code)
        console.print("[green]Remote run finished; outputs downloaded.[/green]")
    else:
        console.print("Plan-only mode: re-run with [bold]--cloud-submit[/bold] to launch.")


@app.command("evaluate")
def evaluate(
    run_dir: str = typer.Argument(...),
    dataset_version: str = typer.Option(..., "--dataset", "-d"),
    root: str = typer.Option("learn", "--root"),
) -> None:
    """Evaluate a downloaded checkpoint (real metrics from a real model)."""
    report = _workflow(root).evaluate_checkpoint(run_dir=run_dir, dataset_version=dataset_version)
    console.print_json(json.dumps({k: v for k, v in report.items() if not k.startswith("_")}))


@app.command("register")
def register(
    run_dir: str = typer.Argument(...),
    dataset_version: str = typer.Option(..., "--dataset", "-d"),
    name: str = typer.Option(..., "--name", "-n"),
    tag: str = typer.Option("scratch", "--tag"),
    notes: Optional[str] = typer.Option(None, "--notes"),
    root: str = typer.Option("learn", "--root"),
) -> None:
    """Register the trained checkpoint in the model registry."""
    entry_id = _workflow(root).register_run(
        run_dir=run_dir, dataset_version=dataset_version, name=name, tag=tag, notes=notes
    )
    console.print(f"Registry entry: [bold]{entry_id}[/bold]")


@app.command("status")
def status(root: str = typer.Option("learn", "--root")) -> None:
    """Show the workflow state (examples, datasets, planned runs)."""
    state = _workflow(root)
    counts = state.review()
    console.print(
        Panel(
            f"pending: {len(counts['pending'])}  approved: {len(counts['approved'])}  "
            f"rejected: {len(counts['rejected'])}",
            title=os.path.abspath(root),
        )
    )

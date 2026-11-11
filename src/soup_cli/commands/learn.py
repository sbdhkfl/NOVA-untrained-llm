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
from typer.core import TyperGroup

console = Console()


class _LearnGroup(TyperGroup):
    """Route a bare request to ``start`` — ``soup learn Python`` predates the
    subcommand layout, so anything that is not a subcommand or an option is
    a learning request (``soup learn from this document notes.md``)."""

    def parse_args(self, ctx, args):
        if (
            args
            and args[0] not in self.commands
            and args[0] not in ctx.help_option_names
            and not args[0].startswith("-")
        ):
            args = ["start", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=_LearnGroup,
    help=(
        "Learn a language model from scratch: `soup learn <topic>` or "
        "`soup learn from this document|dataset <path>` to collect material, "
        "then review/approve, train REMOTELY (Modal) and register the result."
    ),
    no_args_is_help=True,
)


def _workflow(root: str):
    from soup_cli.scratch.learn import LearnWorkflow

    return LearnWorkflow(root)


@app.command("start")
def start(
    request: list[str] = typer.Argument(
        ...,
        help=(
            'Learning request: "Python", "how computers work", '
            '"from this document notes.md", "from this dataset data.jsonl".'
        ),
    ),
    material: Optional[str] = typer.Option(
        None, "--from", help="File or directory of learning material for a topic."
    ),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
) -> None:
    """Start learning from a free-form request (material lands in pending)."""
    from soup_cli.scratch.learn import parse_learn_request

    try:
        req = parse_learn_request(request)
    except ValueError as exc:
        console.print(f"[red]invalid request:[/] {exc}")
        raise typer.Exit(code=2) from exc
    wf = _workflow(root)
    try:
        result = wf.ingest_request(req, material=material)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    for source, count in result["sources"].items():
        console.print(f"  [dim]{source}:[/dim] {count} example(s)")
    console.print(
        f"[green]Ingested {result['examples_ingested']} example(s) "
        f"for {req.topic or req.source}[/green] — all pending, none trained on yet."
    )
    console.print(
        "Next: [bold]soup learn review[/bold] → [bold]soup learn approve[/bold] → "
        "[bold]soup learn prepare[/bold] → [bold]soup learn train[/bold]"
    )


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
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
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
    example_ids: Optional[list[str]] = typer.Argument(None, help="Example ids to approve."),
    all_pending: bool = typer.Option(False, "--all", help="Approve every pending example."),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
) -> None:
    """Explicitly approve examples for training (never automatic)."""
    wf = _workflow(root)
    if not all_pending and not example_ids:
        raise typer.BadParameter("give example ids or --all")
    try:
        approved = wf.approve(example_ids, all_pending=all_pending)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Approved {len(approved)} examples.[/green]")


@app.command("reject")
def reject(
    example_ids: list[str] = typer.Argument(..., help="Example ids to reject."),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
) -> None:
    """Reject pending examples (they will never be trained on)."""
    try:
        rejected = _workflow(root).reject(list(example_ids))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Rejected {len(rejected)} examples.")


@app.command("edit")
def edit(
    example_id: str = typer.Argument(..., help="Example id (see `soup learn review`)."),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Replacement text."),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="File with replacement text."),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
) -> None:
    """Edit an example; the edited example returns to pending for re-review."""
    if bool(text) == bool(file):
        raise typer.BadParameter("give exactly one of --text or --file")
    if file:
        from soup_cli.utils.paths import is_under_cwd

        if not is_under_cwd(file):
            raise typer.BadParameter(f"file must stay under the current working directory: {file}")
        with open(file, encoding="utf-8") as fh:
            text = fh.read()
    assert text is not None
    try:
        example = _workflow(root).edit_example(example_id, text)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Edited {example_id}[/green] → new id {example['example_id']} "
        "(status: pending — re-approve before it can be trained on)."
    )


@app.command("prepare")
def prepare(
    version: str = typer.Option(..., "--version", "-v", help="Dataset version tag."),
    vocab_size: int = typer.Option(512, "--vocab-size", min=260, help="Tokenizer vocabulary size."),
    val_fraction: float = typer.Option(
        0.1, "--val-fraction", min=0.01, max=0.5, help="Validation split fraction."
    ),
    seed: int = typer.Option(1337, "--seed", help="Split shuffle seed."),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
) -> None:
    """Tokenise APPROVED examples into a versioned train/val dataset."""
    try:
        manifest = _workflow(root).prepare_dataset(
            version=version, vocab_size=vocab_size, val_fraction=val_fraction, seed=seed
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(manifest["split"]))


@app.command("train")
def train(
    dataset_version: str = typer.Option(..., "--dataset", "-d", help="Prepared dataset version."),
    gpu: str = typer.Option("a10g", "--gpu", help="Modal GPU type."),
    submit: bool = typer.Option(False, "--cloud-submit", help="Actually submit the remote run."),
    epochs: int = typer.Option(3, "--epochs", min=1, help="Training epochs."),
    batch_size: int = typer.Option(8, "--batch-size", min=1, help="Batch size."),
    learning_rate: float = typer.Option(3e-4, "--lr", min=1e-8, help="Learning rate."),
    n_layers: int = typer.Option(4, "--layers", min=1, help="Transformer layers."),
    n_heads: int = typer.Option(4, "--heads", min=1, help="Attention heads."),
    d_model: int = typer.Option(128, "--d-model", min=8, help="Model width."),
    block_size: int = typer.Option(256, "--block-size", min=16, help="Context length."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Remote run name."),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
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
    try:
        plan = wf.plan_remote_training(
            dataset_version=dataset_version,
            model_config=model_config,
            training_config=training_config,
            gpu=gpu,
            run_name=run_name,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]training plan failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(Panel(plan.stub_text[:800] + "\n…", title="Modal stub (plan-only)"))
    console.print(f"Stub written: [bold]{plan.stub_path}[/bold]")
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
        console.print(
            "Progress streams to this terminal while `modal run` executes; "
            "check [bold]soup learn status[/bold] afterwards."
        )


@app.command("evaluate")
def evaluate(
    run_dir: str = typer.Argument(..., help="Downloaded run dir (or checkpoint dir)."),
    dataset_version: str = typer.Option(..., "--dataset", "-d", help="Dataset version to eval on."),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
) -> None:
    """Evaluate a downloaded checkpoint (real metrics from a real model)."""
    try:
        report = _workflow(root).evaluate_checkpoint(
            run_dir=run_dir, dataset_version=dataset_version
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]evaluation failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps({k: v for k, v in report.items() if not k.startswith("_")}))


@app.command("register")
def register(
    run_dir: str = typer.Argument(..., help="Downloaded run dir (or checkpoint dir)."),
    dataset_version: str = typer.Option(..., "--dataset", "-d", help="Dataset version trained on."),
    name: str = typer.Option(..., "--name", "-n", help="Registry entry name."),
    tag: str = typer.Option("scratch", "--tag", help="Registry tag."),
    notes: Optional[str] = typer.Option(None, "--notes", help="Free-form notes."),
    root: str = typer.Option("learn", "--root", help="Workflow root directory."),
) -> None:
    """Register the trained checkpoint in the model registry."""
    try:
        entry_id = _workflow(root).register_run(
            run_dir=run_dir, dataset_version=dataset_version, name=name, tag=tag, notes=notes
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]registration failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Registry entry: [bold]{entry_id}[/bold]")


@app.command("generate")
def generate(
    run_dir: str = typer.Argument(..., help="Downloaded run dir (or checkpoint dir)."),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to continue."),
    max_tokens: int = typer.Option(64, "--max-tokens", min=1, max=4096, help="Tokens to generate."),
    temperature: float = typer.Option(
        0.0, "--temperature", min=0.0, max=2.0, help="Sampling temperature (0 = greedy)."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
) -> None:
    """Generate text with a trained checkpoint (inference, no training)."""
    from soup_cli.scratch.inference import generate_from_run

    try:
        output = generate_from_run(
            run_dir,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(Panel(output or "(empty — the model emitted only special tokens)"))


@app.command("models")
def models(
    limit: int = typer.Option(50, "--limit", "-l", min=1, max=1000, help="Maximum entries."),
) -> None:
    """List registered models produced by `soup learn`."""
    from soup_cli.registry.store import RegistryStore

    entries = RegistryStore().list(task="scratch", limit=limit)
    if not entries:
        console.print("No registered models yet — run [bold]soup learn register[/bold] first.")
        return
    table = Table(title="registered from-scratch models")
    table.add_column("name")
    table.add_column("tag")
    table.add_column("id")
    table.add_column("created")
    table.add_column("notes")
    for entry in entries:
        table.add_row(
            entry.get("name", ""),
            ", ".join(entry.get("tags", [])),
            entry.get("id", ""),
            entry.get("created_at", ""),
            (entry.get("notes") or "")[:40],
        )
    console.print(table)


@app.command("status")
def status(root: str = typer.Option("learn", "--root", help="Workflow root directory.")) -> None:
    """Show workflow state: examples, datasets, remote-run progress, models."""
    state = _workflow(root).status()
    examples = state["examples"]
    console.print(
        Panel(
            f"pending: {examples['pending']}  approved: {examples['approved']}  "
            f"rejected: {examples['rejected']}",
            title=os.path.abspath(root),
        )
    )
    if state["datasets"]:
        table = Table(title="datasets")
        table.add_column("version")
        table.add_column("train tokens")
        table.add_column("val tokens")
        for ds in state["datasets"]:
            split = ds["split"]
            table.add_row(
                ds["version"],
                str(split.get("train_tokens", "?")),
                str(split.get("val_tokens", "?")),
            )
        console.print(table)
    if state["plans"]:
        table = Table(title="remote training runs (history)")
        table.add_column("run")
        table.add_column("dataset")
        table.add_column("gpu")
        table.add_column("progress")
        table.add_column("eval")
        for plan in state["plans"]:
            table.add_row(
                plan["name"],
                plan.get("dataset_version", ""),
                plan.get("gpu", ""),
                plan["progress"],
                "yes" if plan["eval_report"] else "-",
            )
        console.print(table)
    else:
        console.print("No training plans yet — [bold]soup learn train --dataset <v>[/bold].")
    if state["models"]:
        table = Table(title="registered models")
        table.add_column("name")
        table.add_column("tag")
        table.add_column("id")
        for entry in state["models"]:
            table.add_row(
                entry.get("name", ""),
                ", ".join(entry.get("tags", [])),
                entry.get("id", ""),
            )
        console.print(table)

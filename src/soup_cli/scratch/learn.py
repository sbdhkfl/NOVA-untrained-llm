"""The ``soup learn`` workflow orchestrator.

Wires the full learning-to-training pipeline:

    ingest -> review -> approve -> tokenize+split -> REMOTE train -> eval -> register

Nothing trains locally. The train stage renders a Modal plan (plan-only by
default; ``--cloud-submit`` to actually submit) and every failure mode is
reported loudly — there is no fallback to local training.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console

console = Console()

DEFAULT_LEARN_ROOT = "learn"


@dataclass
class LearnPaths:
    """On-disk layout for the learn workflow under one root."""

    root: str

    @property
    def dataset_root(self) -> str:
        return os.path.join(self.root, "data")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.root, "runs")

    @property
    def learn_state(self) -> str:
        return os.path.join(self.root, "learn_state.json")

    def ensure(self) -> None:
        os.makedirs(self.dataset_root, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)


class LearnWorkflow:
    """High-level orchestration for ``soup learn``."""

    def __init__(self, root: str = DEFAULT_LEARN_ROOT):
        self.paths = LearnPaths(root)
        self.paths.ensure()

    # ------------------------------------------------------------------ #
    # stages 1-3: ingest / review / approve
    # ------------------------------------------------------------------ #

    def _store(self) -> Any:
        from soup_cli.scratch.dataset import ScratchDatasetStore

        return ScratchDatasetStore(self.paths.dataset_root)

    def ingest_text(self, text: str, *, source: str = "user") -> dict[str, Any]:
        example = self._store().ingest_text(text, source=source)
        return example.to_dict()

    def ingest_file(self, path: str, *, source: str | None = None) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._store().ingest_file(path, source=source)]

    def review(self) -> dict[str, Any]:
        store = self._store()
        return {
            "pending": [e.to_dict() for e in store.list_pending()],
            "approved": [e.to_dict() for e in store.list_approved()],
            "rejected": [e.to_dict() for e in store.list_rejected()],
        }

    def approve(
        self, example_ids: list[str] | None = None, *, all_pending: bool = False
    ) -> list[dict[str, Any]]:
        store = self._store()
        if all_pending:
            approved = store.approve_all()
        else:
            if not example_ids:
                raise ValueError("no example ids given and --all not set")
            approved = [store.approve(i) for i in example_ids]
        store.dedupe()
        return [e.to_dict() for e in approved]

    def reject(self, example_ids: list[str]) -> list[dict[str, Any]]:
        store = self._store()
        return [e.to_dict() for e in (store.reject(i) for i in example_ids)]

    # ------------------------------------------------------------------ #
    # stage 4: tokenize + split (approval-gated)
    # ------------------------------------------------------------------ #

    def prepare_dataset(
        self,
        *,
        version: str,
        vocab_size: int = 512,
        val_fraction: float = 0.1,
        seed: int = 1337,
    ) -> dict[str, Any]:
        """Train the tokenizer from approved material and build the split."""
        from soup_cli.scratch.dataset import ScratchDatasetStore
        from soup_cli.scratch.tokenizer import ScratchTokenizer

        store = ScratchDatasetStore(self.paths.dataset_root)
        approved = store.list_approved()
        if not approved:
            raise RuntimeError(
                "no approved examples — refusing to build a dataset. "
                "Approve examples with `soup learn approve` first."
            )
        tokenizer = ScratchTokenizer(vocab_size=vocab_size)
        tokenizer.train([e.text for e in approved])
        manifest = store.build_dataset(
            tokenizer,
            version=version,
            val_fraction=val_fraction,
            seed=seed,
        )
        console.print(
            f"[green]dataset {version} ready:[/green] "
            f"{manifest['split']['train_tokens']} train tokens, "
            f"{manifest['split']['val_tokens']} val tokens"
        )
        return manifest

    # ------------------------------------------------------------------ #
    # stage 5: REMOTE training (plan-only by default)
    # ------------------------------------------------------------------ #

    def plan_remote_training(
        self,
        *,
        dataset_version: str,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        gpu: str = "a10g",
        output_dir: str | None = None,
        soup_version: str | None = None,
        run_name: str | None = None,
    ) -> Any:
        """Render the Modal training plan. NEVER trains locally. Returns the
        :class:`CloudPlan` (stub text + planned ``modal run`` command)."""
        from soup_cli.scratch import remote as scratch_remote
        from soup_cli.scratch.dataset import ScratchDatasetStore

        if output_dir is None:
            output_dir = os.path.join(self.paths.runs_dir, f"{dataset_version}-{run_name or 'run'}")
        store = ScratchDatasetStore(self.paths.dataset_root)
        payload = scratch_remote.build_payload_from_store(
            store,
            dataset_version=dataset_version,
            model_config=model_config,
            training_config=training_config,
        )
        from soup_cli import __version__ as soup_version_or_none

        version = soup_version or soup_version_or_none or "0.0.0.dev0"
        plan = scratch_remote.plan_scratch_modal_run(
            payload.to_bundle(),
            gpu=gpu,
            output_dir=output_dir,
            soup_version=version,
            run_name=run_name,
        )
        # Record the plan for reproducibility.
        state = self._load_state()
        state.setdefault("plans", {})[run_name or dataset_version] = {
            "stub_path": plan.stub_path,
            "run_command": plan.run_command,
            "gpu": plan.gpu,
            "output_dir": plan.output_dir,
            "dataset_version": dataset_version,
            "model_config": model_config,
            "training_config": training_config,
            "planned_at": time.time(),
        }
        self._save_state(state)
        return plan

    # ------------------------------------------------------------------ #
    # stage 6: local evaluation of a downloaded checkpoint
    # ------------------------------------------------------------------ #

    def evaluate_checkpoint(self, *, run_dir: str, dataset_version: str) -> dict[str, Any]:
        """Evaluate a checkpoint that was downloaded from the remote run.
        Requires the checkpoint to already exist locally."""
        from soup_cli.scratch import eval as scratch_eval
        from soup_cli.scratch.dataset import ScratchDatasetStore
        from soup_cli.scratch.engine import CheckpointManager
        from soup_cli.scratch.model import ScratchModelConfig, ScratchTransformer
        from soup_cli.scratch.tokenizer import ScratchTokenizer

        loaded = ScratchDatasetStore(self.paths.dataset_root).load_dataset(dataset_version)
        ckpt = CheckpointManager(os.path.join(run_dir, "checkpoint-final"))
        state = ckpt.load()
        model_cfg = ScratchModelConfig.from_dict(state["model_config"])
        model = ScratchTransformer(model_cfg).build()
        ckpt.restore_model(model)
        ckpt_tok_path = os.path.join(run_dir, "checkpoint-final", "tokenizer.json")
        ScratchTokenizer.load(ckpt_tok_path)  # verify tokenizer compatibility

        report = scratch_eval.evaluate_model(
            model=model,
            val_ids=loaded["val_ids"],
            block_size=model_cfg.block_size,
        )
        report.update(scratch_eval.dataset_statistics(loaded["manifest"]))
        report["run_dir"] = run_dir
        report["checkpoint_step"] = state.get("step")
        path = scratch_eval.write_report(report, os.path.join(run_dir, "eval_report.json"))
        console.print(
            f"[green]eval:[/green] val_loss={report['val_loss']:.4f} "
            f"perplexity={report['val_perplexity']:.2f}"
        )
        report["_path"] = path
        return report

    # ------------------------------------------------------------------ #
    # stage 7: registry integration
    # ------------------------------------------------------------------ #

    def register_run(
        self,
        *,
        run_dir: str,
        dataset_version: str,
        name: str,
        tag: str = "scratch",
        notes: str | None = None,
    ) -> str:
        """Register the trained checkpoint in the model registry. Only a run
        with a real ``run_report.json`` from the remote worker qualifies."""
        from soup_cli.registry.store import RegistryStore

        report_path = os.path.join(run_dir, "run_report.json")
        if not os.path.isfile(report_path):
            raise RuntimeError(
                f"no run_report.json in {run_dir}: refusing to register a run "
                "without real remote training evidence"
            )
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
        store = RegistryStore()
        entry_id = store.push(
            name=name,
            tag=tag,
            base_model=(
                f"scratch:{report['model_config'].get('architecture', 'causal-transformer')}"
            ),
            task="scratch",
            run_id=None,
            config={
                "model": report["model_config"],
                "training": report["training_config"],
                "dataset": report["dataset_manifest"],
                "metrics": {
                    "final_train_loss": report["final_train_loss"],
                    "final_val_loss": report["final_val_loss"],
                    "final_val_perplexity": report["final_val_perplexity"],
                },
            },
            data_path=os.path.join(
                self.paths.dataset_root, "datasets", dataset_version, "manifest.json"
            ),
            notes=notes,
        )
        artifact_path = os.path.join(run_dir, "checkpoint-final")
        if os.path.isdir(artifact_path):
            store.add_artifact(
                entry_id,
                kind="checkpoint",
                path=os.path.relpath(artifact_path),
            )
        console.print(f"[green]registered:[/green] {entry_id}")
        return entry_id

    # ------------------------------------------------------------------ #

    def _load_state(self) -> dict[str, Any]:
        if os.path.isfile(self.paths.learn_state):
            with open(self.paths.learn_state, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        with open(self.paths.learn_state, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

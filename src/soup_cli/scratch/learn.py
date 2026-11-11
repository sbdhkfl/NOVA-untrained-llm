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
import re
import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console

console = Console()

DEFAULT_LEARN_ROOT = "learn"


def resolve_checkpoint_dir(run_dir: str) -> str:
    """Resolve a run directory (or checkpoint directory) to the directory
    holding ``training_state.json``. Accepts either ``<run>/`` containing
    ``checkpoint-final/``, or the checkpoint directory itself."""
    candidates = [
        os.path.join(run_dir, "checkpoint-final"),
        run_dir,
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "training_state.json")):
            return candidate
    raise FileNotFoundError(
        f"no checkpoint found under {run_dir} (expected checkpoint-final/"
        "training_state.json): download the remote run outputs first"
    )


# --------------------------------------------------------------------------- #
# free-form learning requests: `soup learn Python`,
# `soup learn from this document notes.md`, `soup learn from this dataset d.jsonl`
# --------------------------------------------------------------------------- #

_DATASET_EXTENSIONS = frozenset({".jsonl", ".json", ".csv", ".parquet"})
_MATERIAL_EXTENSIONS = frozenset({".txt", ".md"} | _DATASET_EXTENSIONS)


@dataclass(frozen=True)
class LearnRequest:
    """A parsed learning request."""

    kind: str  # "topic" | "document" | "dataset"
    topic: str = ""
    source: str = ""


def parse_learn_request(words: "list[str] | tuple[str, ...]") -> LearnRequest:
    """Parse the free-form ``soup learn ...`` argument into a request.

    Accepts a topic (``Python``, ``how computers work``) or the fixed shapes
    ``from [this] document <path>`` / ``from [this] dataset <path>``.
    """
    parts = [w for w in words if w.strip()]
    if not parts:
        raise ValueError(
            'empty request — try `soup learn Python` or '
            '`soup learn from this document notes.md`'
        )
    if parts[0] == "from":
        rest = parts[1:]
        if rest and rest[0] == "this":
            rest = rest[1:]
        if len(rest) >= 2 and rest[0] in ("document", "dataset"):
            return LearnRequest(kind=rest[0], source=" ".join(rest[1:]))
        raise ValueError(
            'expected `soup learn from this document <path>` or '
            '`soup learn from this dataset <path>`'
        )
    return LearnRequest(kind="topic", topic=" ".join(parts))


def topic_slug(topic: str) -> str:
    """Filesystem-safe slug for a topic name."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug or "topic"


def _looks_like_dataset(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _DATASET_EXTENSIONS


def _resolve_topic_material(
    topic: str, *, root: str, material: str | None
) -> "list[str]":
    """Locate material for a topic: explicit ``--from`` first, then
    ``<root>/material/<slug>[/<files>]`` and ``<root>/material/<slug>.<ext>``."""
    if material:
        if os.path.isdir(material):
            return _material_files(material)
        return [material]
    slug = topic_slug(topic)
    mat_dir = os.path.join(root, "material", slug)
    if os.path.isdir(mat_dir):
        files = _material_files(mat_dir)
        if files:
            return files
    for ext in sorted(_MATERIAL_EXTENSIONS):
        candidate = os.path.join(root, "material", slug + ext)
        if os.path.isfile(candidate):
            return [candidate]
    return []


def _material_files(directory: str) -> "list[str]":
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if os.path.splitext(name)[1].lower() in _MATERIAL_EXTENSIONS
        and os.path.isfile(os.path.join(directory, name))
    ]


def row_to_text(row: Any) -> str:
    """Extract training text from one dataset row."""
    if not isinstance(row, dict):
        return str(row)
    instruction, output = row.get("instruction"), row.get("output")
    if (
        isinstance(instruction, str)
        and isinstance(output, str)
        and instruction.strip()
        and output.strip()
    ):
        return f"{instruction.strip()}\n{output.strip()}"
    for key in ("text", "content", "completion", "answer", "chosen"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("messages", "conversations"):
        messages = row.get(key)
        if isinstance(messages, list):
            parts = [
                m.get("content")
                for m in messages
                if isinstance(m, dict) and isinstance(m.get("content"), str)
            ]
            if parts:
                return "\n".join(parts)
    strings = [v for v in row.values() if isinstance(v, str) and v.strip()]
    if strings:
        return "\n".join(strings)
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _as_path(path: str) -> Any:
    from pathlib import Path

    return Path(path)


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

    def edit_example(self, example_id: str, new_text: str) -> dict[str, Any]:
        """Replace an example's text; the result returns to ``pending`` for
        re-review."""
        return self._store().edit(example_id, new_text).to_dict()

    # ------------------------------------------------------------------ #
    # stage 1: free-form learning requests (`soup learn <request>`)
    # ------------------------------------------------------------------ #

    def ingest_request(
        self, request: "LearnRequest", *, material: str | None = None
    ) -> dict[str, Any]:
        """Collect learning material for a parsed request and ingest it into
        the pending bucket. Nothing is approved here — approval is always
        explicit, in a later step."""
        if request.kind == "document":
            sources = [request.source]
        elif request.kind == "dataset":
            sources = [request.source]
        else:  # topic
            sources = _resolve_topic_material(
                request.topic, root=self.paths.root, material=material
            )
            if not sources:
                raise FileNotFoundError(
                    f"no learning material found for topic {request.topic!r}. Provide it with "
                    f"--from <file-or-dir>, or place files under "
                    f"{os.path.join(self.paths.root, 'material', topic_slug(request.topic))}/ "
                    '(supported: .txt, .md, or a dataset as .jsonl/.json/.csv)'
                )
        total = 0
        per_source: dict[str, int] = {}
        for source in sources:
            examples = (
                self.ingest_dataset(source)
                if _looks_like_dataset(source)
                else self.ingest_file(source)
            )
            per_source[source] = len(examples)
            total += len(examples)
        return {
            "kind": request.kind,
            "topic": request.topic,
            "sources": per_source,
            "examples_ingested": total,
        }

    def ingest_dataset(self, path: str) -> list[dict[str, Any]]:
        """Ingest a dataset file (jsonl / json / csv / parquet): one pending
        example per row, with the row's text extracted."""
        from soup_cli.data.loader import load_raw_data

        rows = load_raw_data(_as_path(path))
        examples = []
        for row in rows:
            text = row_to_text(row)
            if text.strip():
                examples.append(self.ingest_text(text, source=os.path.basename(path)))
        return examples

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
        # The dataset tokenizer is the source of truth for the vocabulary:
        # a model sized below it would index out of range on remote GPU.
        tok_vocab = int(payload.dataset_manifest.get("tokenizer", {}).get("vocab_size") or 0)
        if tok_vocab and model_config.get("vocab_size") != tok_vocab:
            model_config = {**model_config, "vocab_size": tok_vocab}
            payload.model_config = model_config
        from soup_cli import __version__ as soup_version_or_none

        version = soup_version or soup_version_or_none or "0.0.0.dev0"
        plan = scratch_remote.plan_scratch_modal_run(
            payload.to_bundle(),
            gpu=gpu,
            output_dir=output_dir,
            soup_version=version,
            run_name=run_name,
        )
        # The stub must exist on disk: `modal run <stub>` and the plan-only
        # printed command both reference it.
        stub_realpath = scratch_remote.write_scratch_stub(plan)
        # Record the plan for reproducibility.
        state = self._load_state()
        state.setdefault("plans", {})[run_name or dataset_version] = {
            "stub_path": stub_realpath,
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
        ckpt_dir = resolve_checkpoint_dir(run_dir)
        ckpt = CheckpointManager(ckpt_dir)
        state = ckpt.load()
        model_cfg = ScratchModelConfig.from_dict(state["model_config"])
        model = ScratchTransformer(model_cfg).build()
        ckpt.restore_model(model)
        ckpt_tok_path = os.path.join(ckpt_dir, "tokenizer.json")
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
        artifact_path = os.path.join(resolve_checkpoint_dir(run_dir), "model.safetensors")
        if os.path.isfile(artifact_path):
            store.add_artifact(
                entry_id=entry_id,
                kind="checkpoint",
                path=os.path.relpath(artifact_path),
            )
        console.print(f"[green]registered:[/green] {entry_id}")
        return entry_id

    # ------------------------------------------------------------------ #
    # workflow status / history / model versions
    # ------------------------------------------------------------------ #

    def status(self) -> dict[str, Any]:
        """Everything the status view needs: example buckets, dataset
        versions, planned remote runs with their progress, and the
        registered models produced by this workflow."""
        store = self._store()
        examples = {
            name: len(getattr(store, f"list_{name}")())
            for name in ("pending", "approved", "rejected")
        }
        datasets = []
        for version in store.list_versions():
            manifest_path = os.path.join(store.root, "datasets", version, "manifest.json")
            manifest: dict[str, Any] = {}
            if os.path.isfile(manifest_path):
                with open(manifest_path, encoding="utf-8") as fh:
                    manifest = json.load(fh)
            datasets.append({"version": version, "split": manifest.get("split", {})})
        plans = []
        for name, plan in sorted(self._load_state().get("plans", {}).items()):
            output_dir = str(plan.get("output_dir", ""))
            if os.path.isfile(os.path.join(output_dir, "run_report.json")):
                progress = "finished"
            elif os.path.isdir(output_dir):
                progress = "outputs downloaded, no run_report.json (failed/interrupted)"
            else:
                progress = "planned; remote outputs not downloaded yet"
            plans.append(
                {
                    **plan,
                    "name": name,
                    "progress": progress,
                    "eval_report": os.path.isfile(os.path.join(output_dir, "eval_report.json")),
                }
            )
        try:
            from soup_cli.registry.store import RegistryStore

            models = RegistryStore().list(task="scratch", limit=50)
        except Exception:  # noqa: BLE001 — a broken registry must not break status
            models = []
        return {
            "examples": examples,
            "datasets": datasets,
            "plans": plans,
            "models": models,
        }

    # ------------------------------------------------------------------ #

    def _load_state(self) -> dict[str, Any]:
        if os.path.isfile(self.paths.learn_state):
            with open(self.paths.learn_state, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        with open(self.paths.learn_state, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

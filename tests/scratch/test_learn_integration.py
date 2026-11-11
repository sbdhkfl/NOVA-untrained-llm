"""Integration tests for `soup learn` — the mission-critical paths.

Covers: learning-request parsing, dataset/document submission, training-job
submission (via the shared Modal override seam — never a real remote call),
training status retrieval, model registration, model loading, inference,
failure handling, and the end-to-end workflow.

Everything runs on tiny synthetic data with a tiny model: nothing is
downloaded and no training happens outside the explicitly local unit loops
(production training is remote-only via Modal).
"""

from __future__ import annotations

import json
import os

import pytest

from tests.conftest import make_cli_runner

runner = make_cli_runner()

CORPUS = [
    "Python is a programming language. Variables store values, functions group "
    "statements, loops repeat work, and conditions choose paths. Programs read "
    "input, compute results, and write output for people to use.",
    "Mathematics studies numbers, shapes, and patterns. Addition combines "
    "quantities, multiplication scales them, and algebra replaces unknown "
    "values with letters so equations can be solved systematically.",
    "Computers execute instructions stored in memory. The CPU fetches, decodes "
    "and runs each instruction, while storage keeps data and programs when the "
    "power is switched off between sessions.",
    "Databases organise information into tables with rows and columns. Queries "
    "select, insert and update records, and transactions keep the data "
    "consistent even when several users change it at the same time.",
]


def _app():
    from soup_cli.cli import app

    return app


def _prepared_workflow(tmp_path, monkeypatch, *, version: str = "v1"):
    """chdir + ingest + approve + prepare a dataset version (API level)."""
    from soup_cli.scratch.learn import LearnWorkflow

    monkeypatch.chdir(tmp_path)
    wf = LearnWorkflow("learn")
    for text in CORPUS:
        wf.ingest_text(text, source="synthetic")
    wf.approve(all_pending=True)
    wf.prepare_dataset(version=version, vocab_size=300)
    return wf


def _simulate_remote_run(tmp_path, *, dataset_version: str = "v1") -> str:
    """Produce what a finished remote run leaves behind: a trained tiny
    checkpoint plus run_report.json (the remote-training evidence)."""
    pytest.importorskip("torch")
    from soup_cli.scratch.dataset import ScratchDatasetStore
    from soup_cli.scratch.engine import ScratchTrainer, ScratchTrainingConfig
    from soup_cli.scratch.model import ScratchModelConfig, ScratchTransformer

    store = ScratchDatasetStore(str(tmp_path / "learn" / "data"))
    loaded = store.load_dataset(dataset_version)
    model_config = ScratchModelConfig(
        vocab_size=300, block_size=16, n_layers=1, n_heads=2, d_model=16
    )
    model = ScratchTransformer(model_config).build()
    training_config = ScratchTrainingConfig(
        epochs=1,
        batch_size=4,
        block_size=16,
        max_steps=2,
        eval_interval=1000,
        save_interval=1000,
        log_interval=1,
        eval_batches=1,
    )
    run_dir = tmp_path / "learn" / "runs" / "v1-run"
    # Mirror the remote worker: the trainer only copies tokenizer.json into
    # the checkpoint when given tokenizer_path (remote_train.py does this).
    tok_src = tmp_path / "learn" / "data" / "datasets" / dataset_version / "tokenizer.json"
    assert tok_src.is_file(), f"dataset tokenizer missing: {tok_src}"
    trainer = ScratchTrainer(
        model=model,
        train_ids=loaded["train_ids"],
        val_ids=loaded["val_ids"],
        config=training_config,
        checkpoint_dir=str(run_dir / "checkpoint-final"),
        tokenizer_path=str(tok_src),
        model_config=model_config.to_dict(),
    )
    result = trainer.train()
    report = {
        "steps": result.steps,
        "final_train_loss": result.final_train_loss,
        "final_val_loss": result.final_val_loss,
        "final_val_perplexity": result.final_val_perplexity,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "dataset_manifest": loaded["manifest"],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_report.json").write_text(json.dumps(report), encoding="utf-8")
    return str(run_dir)


# --------------------------------------------------------------------------- #
# learning-request parsing
# --------------------------------------------------------------------------- #


def test_parse_topic_request():
    from soup_cli.scratch.learn import parse_learn_request

    req = parse_learn_request(["Python"])
    assert req.kind == "topic" and req.topic == "Python"
    req = parse_learn_request(["how", "computers", "work"])
    assert req.kind == "topic" and req.topic == "how computers work"


def test_parse_document_request():
    from soup_cli.scratch.learn import parse_learn_request

    req = parse_learn_request(["from", "this", "document", "notes.md"])
    assert req.kind == "document" and req.source == "notes.md"
    # "this" is optional; everything after the keyword is the path, so
    # multi-word filenames work
    req = parse_learn_request(["from", "document", "a", "b.md"])
    assert req.kind == "document" and req.source == "a b.md"


def test_parse_dataset_request():
    from soup_cli.scratch.learn import parse_learn_request

    req = parse_learn_request(["from", "this", "dataset", "data.jsonl"])
    assert req.kind == "dataset" and req.source == "data.jsonl"


def test_parse_request_errors():
    from soup_cli.scratch.learn import parse_learn_request

    with pytest.raises(ValueError, match="empty"):
        parse_learn_request([])
    with pytest.raises(ValueError, match="from this"):
        parse_learn_request(["from", "something", "else"])
    with pytest.raises(ValueError, match="from this"):
        parse_learn_request(["from", "document"])  # missing path


def test_topic_slug_is_filesystem_safe():
    from soup_cli.scratch.learn import topic_slug

    assert topic_slug("How Computers Work!") == "how-computers-work"
    assert topic_slug("Python") == "python"


def test_row_to_text_shapes():
    from soup_cli.scratch.learn import row_to_text

    # alpaca rows keep both sides
    assert row_to_text({"instruction": "2+2?", "output": "4"}) == "2+2?\n4"
    # plain text rows
    assert row_to_text({"text": "hello"}) == "hello"
    # chat rows are flattened
    chat = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    assert row_to_text(chat) == "hi\nhello"
    # unknown shapes fall back to every string value / JSON
    assert "x" in row_to_text({"a": "x"})
    assert row_to_text({"n": 1}) == '{"n": 1}'


# --------------------------------------------------------------------------- #
# submission: topic / document / dataset
# --------------------------------------------------------------------------- #


def test_cli_topic_without_material_exits_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(_app(), ["learn", "Python"])
    assert result.exit_code == 2, result.output
    assert "no learning material" in result.output


def test_cli_topic_with_material_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    material = tmp_path / "learn" / "material" / "python"
    material.mkdir(parents=True)
    (material / "basics.txt").write_text(
        "Python holds values in variables.", encoding="utf-8"
    )
    result = runner.invoke(_app(), ["learn", "Python"])
    assert result.exit_code == 0, result.output
    pending = list((tmp_path / "learn" / "data" / "pending").glob("*.json"))
    assert len(pending) == 1


def test_cli_topic_with_from_option(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.md").write_text("# Notes\nMath adds numbers together.", encoding="utf-8")
    result = runner.invoke(_app(), ["learn", "mathematics", "--from", "corpus.md"])
    assert result.exit_code == 0, result.output
    assert list((tmp_path / "learn" / "data" / "pending").glob("*.json"))


def test_cli_from_this_document(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("Computers execute instructions.", encoding="utf-8")
    result = runner.invoke(_app(), ["learn", "from", "this", "document", "notes.md"])
    assert result.exit_code == 0, result.output
    pending = list((tmp_path / "learn" / "data" / "pending").glob("*.json"))
    assert len(pending) == 1


def test_cli_from_this_dataset_submits_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"instruction": "What is 2+2?", "output": "4"},
        {"instruction": "What is 3+3?", "output": "6"},
        {"text": "plain row text"},
    ]
    with open(tmp_path / "data.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    result = runner.invoke(_app(), ["learn", "from", "this", "dataset", "data.jsonl"])
    assert result.exit_code == 0, result.output
    pending = list((tmp_path / "learn" / "data" / "pending").glob("*.json"))
    assert len(pending) == 3


def test_cli_subcommands_still_route_correctly(tmp_path, monkeypatch):
    """`review` must not be swallowed by the free-form router."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(_app(), ["learn", "review"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(_app(), ["learn", "--help"])
    assert result.exit_code == 0
    for sub in ("start", "review", "approve", "train", "generate", "models"):
        assert sub in result.output


# --------------------------------------------------------------------------- #
# review: approve / reject / edit
# --------------------------------------------------------------------------- #


def test_cli_edit_moves_example_back_to_pending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(_app(), ["learn", "ingest", "--text", "original text to edit"])
    assert result.exit_code == 0, result.output
    pending_dir = tmp_path / "learn" / "data" / "pending"
    old_id = next(pending_dir.glob("*.json")).stem
    result = runner.invoke(
        _app(), ["learn", "edit", old_id, "--text", "edited text that is different"]
    )
    assert result.exit_code == 0, result.output
    new_ids = [p.stem for p in pending_dir.glob("*.json")]
    assert old_id not in new_ids
    assert len(new_ids) == 1


def test_cli_approve_unknown_id_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(_app(), ["learn", "ingest", "--text", "some pending text"])
    result = runner.invoke(_app(), ["learn", "approve", "deadbeefdeadbeef"])
    assert result.exit_code == 1, result.output
    assert "no such example" in result.output


def test_cli_prepare_without_approvals_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(_app(), ["learn", "ingest", "--text", "never approved at all"])
    result = runner.invoke(_app(), ["learn", "prepare", "--version", "v1"])
    assert result.exit_code == 1, result.output
    assert "approved" in result.output


def test_cli_edit_unknown_id_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(_app(), ["learn", "edit", "deadbeefdeadbeef", "--text", "x"])
    assert result.exit_code == 1, result.output
    assert "no such example" in result.output


# --------------------------------------------------------------------------- #
# training-job submission (plan + stub + Modal override seam)
# --------------------------------------------------------------------------- #


def test_plan_writes_the_modal_stub(tmp_path, monkeypatch):
    _prepared_workflow(tmp_path, monkeypatch)
    from soup_cli.scratch.engine import ScratchTrainingConfig
    from soup_cli.scratch.learn import LearnWorkflow
    from soup_cli.scratch.model import ScratchModelConfig

    wf = LearnWorkflow("learn")
    plan = wf.plan_remote_training(
        dataset_version="v1",
        model_config=ScratchModelConfig(
            vocab_size=300, block_size=16, n_layers=1, n_heads=2, d_model=16
        ).to_dict(),
        training_config=ScratchTrainingConfig(block_size=16, max_steps=2).to_dict(),
        gpu="t4",
    )
    assert os.path.isfile("soup_learn_modal_app.py"), "plan must write the stub"
    stub_text = open(plan.stub_path, encoding="utf-8").read()
    assert "soup-scratch-remote" in stub_text


def test_stub_source_mounts_volume_and_compiles(tmp_path, monkeypatch):
    """Regression guard: the rendered stub must mount the output volume
    (otherwise checkpoints are lost), commit it, and be valid Python."""
    _prepared_workflow(tmp_path, monkeypatch)
    from soup_cli.scratch.engine import ScratchTrainingConfig
    from soup_cli.scratch.learn import LearnWorkflow
    from soup_cli.scratch.model import ScratchModelConfig

    wf = LearnWorkflow("learn")
    plan = wf.plan_remote_training(
        dataset_version="v1",
        model_config=ScratchModelConfig(
            vocab_size=300, block_size=16, n_layers=1, n_heads=2, d_model=16
        ).to_dict(),
        training_config=ScratchTrainingConfig(block_size=16, max_steps=2).to_dict(),
        gpu="t4",
    )
    compile(plan.stub_text, plan.stub_path, "exec")  # valid Python
    assert 'volumes={"/outputs": volume}' in plan.stub_text
    assert "volume.commit()" in plan.stub_text
    assert "modal volume get" in plan.stub_text  # manual retry hint
    assert "batch_download" not in plan.stub_text  # unverified API, removed
    # payload model vocab matches the dataset tokenizer (300, not the 256 default)
    # The training payload is base64-embedded; decode it and verify the
    # vocab fix rode along into the remote model config.
    import ast
    import base64
    import re

    m = re.search(r"^PAYLOAD_B64 = (.+)$", plan.stub_text, re.MULTILINE)
    assert m, "stub must embed PAYLOAD_B64"
    payload = json.loads(base64.b64decode(ast.literal_eval(m.group(1))))
    assert payload["format"] == "soup-scratch-remote"
    assert payload["model_config"]["vocab_size"] == 300  # synced to tokenizer


def test_training_job_submission_via_override_seam(tmp_path, monkeypatch):
    """Submit through the shared Modal seam — the job reaches the plan,
    never a real remote call, and never local training."""
    import soup_cli.cloud.modal as modal_mod

    _prepared_workflow(tmp_path, monkeypatch)

    captured = {}

    def fake_submit(plan):
        captured["plan"] = plan
        return 0

    monkeypatch.setattr(modal_mod, "_MODAL_SUBMIT_OVERRIDE", fake_submit)
    result = runner.invoke(
        _app(),
        [
            "learn",
            "train",
            "--dataset",
            "v1",
            "--gpu",
            "t4",
            "--block-size",
            "16",
            "--d-model",
            "16",
            "--layers",
            "1",
            "--cloud-submit",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "plan" in captured
    assert os.path.isfile(captured["plan"].stub_path)


def test_training_job_submission_failure_is_loud(tmp_path, monkeypatch):
    """A failing remote submit must surface a non-zero exit, never a local
    fallback."""
    import soup_cli.cloud.modal as modal_mod

    _prepared_workflow(tmp_path, monkeypatch)

    def failing_submit(plan):
        raise RuntimeError("Modal not authenticated (test)")

    monkeypatch.setattr(modal_mod, "_MODAL_SUBMIT_OVERRIDE", failing_submit)
    result = runner.invoke(
        _app(),
        [
            "learn",
            "train",
            "--dataset",
            "v1",
            "--gpu",
            "t4",
            "--block-size",
            "16",
            "--d-model",
            "16",
            "--layers",
            "1",
            "--cloud-submit",
        ],
    )
    assert result.exit_code != 0
    assert "Modal not authenticated" in str(result.exception)


# --------------------------------------------------------------------------- #
# training status retrieval
# --------------------------------------------------------------------------- #


def test_status_reports_datasets_and_plan_progress(tmp_path, monkeypatch):
    _prepared_workflow(tmp_path, monkeypatch)
    from soup_cli.scratch.engine import ScratchTrainingConfig
    from soup_cli.scratch.learn import LearnWorkflow
    from soup_cli.scratch.model import ScratchModelConfig

    wf = LearnWorkflow("learn")
    wf.plan_remote_training(
        dataset_version="v1",
        model_config=ScratchModelConfig(
            vocab_size=300, block_size=16, n_layers=1, n_heads=2, d_model=16
        ).to_dict(),
        training_config=ScratchTrainingConfig(block_size=16, max_steps=2).to_dict(),
        gpu="t4",
    )
    result = runner.invoke(_app(), ["learn", "status"])
    assert result.exit_code == 0, result.output
    assert "v1" in result.output
    assert "planned; remote outputs not downloaded yet" in result.output
    # once the remote outputs land, status flips to finished
    run_dir = tmp_path / "learn" / "runs" / "v1-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_report.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(_app(), ["learn", "status"])
    assert "finished" in result.output


# --------------------------------------------------------------------------- #
# end to end: evaluate → register → models → generate (inference)
# --------------------------------------------------------------------------- #


def test_full_workflow_evaluate_register_models_generate(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _prepared_workflow(tmp_path, monkeypatch)
    _simulate_remote_run(tmp_path)

    # evaluate the downloaded checkpoint — real forward passes, real metrics
    result = runner.invoke(
        _app(), ["learn", "evaluate", "learn/runs/v1-run", "--dataset", "v1"]
    )
    assert result.exit_code == 0, result.output
    assert os.path.isfile(os.path.join("learn", "runs", "v1-run", "eval_report.json"))
    payload = json.loads(
        open(
            os.path.join("learn", "runs", "v1-run", "eval_report.json"), encoding="utf-8"
        ).read()
    )
    assert payload["val_loss"] > 0 and payload["val_perplexity"] >= 1.0

    # register (requires the run_report.json remote-training evidence)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    result = runner.invoke(
        _app(),
        [
            "learn",
            "register",
            "learn/runs/v1-run",
            "--dataset",
            "v1",
            "--name",
            "demo-scratch",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Registry entry" in result.output

    # the entry exists with its checkpoint artifact
    from soup_cli.registry.store import RegistryStore

    entries = RegistryStore().list(task="scratch")
    assert len(entries) == 1
    assert entries[0]["name"] == "demo-scratch"
    artifacts = RegistryStore().get_artifacts(entries[0]["id"])
    assert any(a["kind"] == "checkpoint" for a in artifacts)

    # model versions are visible through the learn UX
    result = runner.invoke(_app(), ["learn", "models"])
    assert result.exit_code == 0, result.output
    assert "demo-scratch" in result.output

    # inference: load the checkpoint and generate a continuation
    result = runner.invoke(
        _app(),
        [
            "learn",
            "generate",
            "learn/runs/v1-run",
            "--prompt",
            "python",
            "--max-tokens",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output


def test_generate_without_checkpoint_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "empty_run").mkdir()
    result = runner.invoke(
        _app(), ["learn", "generate", "empty_run", "--prompt", "hi"]
    )
    assert result.exit_code == 1, result.output
    assert "no checkpoint" in result.output


def test_register_without_remote_evidence_fails_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    (tmp_path / "fake_run" / "checkpoint-final").mkdir(parents=True)
    result = runner.invoke(
        _app(),
        ["learn", "register", "fake_run", "--dataset", "v1", "--name", "nope"],
    )
    assert result.exit_code == 1, result.output
    assert "run_report" in result.output

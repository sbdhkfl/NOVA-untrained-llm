"""Tests for the from-scratch training subsystem (soup learn).

All tests use small synthetic data and a tiny model; nothing downloads
weights and no training runs outside the explicitly-marked local loop
(which is only exercised here for unit verification — production training
is remote-only via Modal).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.scratch.dataset import (
    ScratchDatasetStore,
    clean_text,
    validate_text,
)
from soup_cli.scratch.engine import ScratchTrainer, ScratchTrainingConfig
from soup_cli.scratch.eval import dataset_statistics, evaluate_model
from soup_cli.scratch.model import ScratchModelConfig, ScratchTransformer
from soup_cli.scratch.tokenizer import ScratchTokenizer


@pytest.fixture
def corpus():
    texts = [
        "the cat sat on the mat and the cat saw the dog",
        "the dog ran to the park and the dog saw the cat",
        "soup learns from approved examples only",
        "approved examples build the tokenised dataset",
    ] * 5
    return texts


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


def test_tokenizer_roundtrip_byte_level():
    tok = ScratchTokenizer(vocab_size=260)
    tok.train(["hello world", "second line of text"])
    text = "hello world — héllo ünïcode ✓"
    ids = tok.encode(text)
    assert tok.decode(ids) == text  # byte-level BPE is lossless


def test_tokenizer_special_tokens():
    tok = ScratchTokenizer(vocab_size=280)
    tok.train(["some corpus text for the tokenizer", "more corpus text"])
    ids = tok.encode("hi", add_special_tokens=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.pad_id != tok.bos_id
    # special tokens are skipped on decode
    assert tok.decode(ids) == "hi"


def test_tokenizer_requires_min_vocab():
    tok = ScratchTokenizer(vocab_size=256)
    with pytest.raises(ValueError):
        tok.train(["abc"])


def test_tokenizer_save_load_roundtrip(tmp_path, corpus):
    tok = ScratchTokenizer(vocab_size=280)
    tok.train(corpus)
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    loaded = ScratchTokenizer.load(path)
    assert loaded.version_hash == tok.version_hash
    text = "the cat sat"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(tok.encode(text)) == text


def test_tokenizer_padding():
    tok = ScratchTokenizer(vocab_size=280)
    tok.train(["a b c", "d e f"])
    batch = tok.pad_batch([[1, 2, 3], [4]], max_len=5)
    assert len(batch[0]) == 5 == len(batch[1])
    assert batch[1][1:] == [tok.pad_id] * 4


def test_tokenizer_stats():
    tok = ScratchTokenizer(vocab_size=300)
    tok.train(["some text for stats", "more stats text here"])
    stats = tok.token_stats()
    assert stats["vocab_size"] == 300
    assert stats["special_tokens"] == 4
    # BPE stops early when no pair occurs >= 2 times, so merges <= budget
    assert 0 < stats["merges"] <= 300 - 260


# --------------------------------------------------------------------------- #
# dataset pipeline
# --------------------------------------------------------------------------- #


def test_clean_and_validate():
    cleaned = clean_text("  hello\x00\tworld\n\n\n\nagain  ")
    assert "\x00" not in cleaned
    assert validate_text(cleaned)["passed"] is True
    assert validate_text("")["passed"] is False


def test_ingest_lands_in_pending_not_approved(tmp_path):
    store = ScratchDatasetStore(str(tmp_path / "store"))
    ex = store.ingest_text("an example awaiting review", source="unit-test")
    assert ex.status == "pending"
    assert len(store.list_pending()) == 1
    assert len(store.list_approved()) == 0
    # explicit approval is required
    approved = store.approve(ex.example_id)
    assert approved.status == "approved"
    assert len(store.list_pending()) == 0
    assert len(store.list_approved()) == 1


def test_invalid_example_is_rejected_not_pending(tmp_path):
    store = ScratchDatasetStore(str(tmp_path / "store"))
    ex = store.ingest_text("")
    assert ex.status == "rejected"
    assert len(store.list_pending()) == 0


def test_approval_gate_blocks_dataset_build(tmp_path):
    store = ScratchDatasetStore(str(tmp_path / "store"))
    store.ingest_text("this was never approved")
    tok = ScratchTokenizer(vocab_size=280)
    tok.train(["filler corpus for the gate test"])
    with pytest.raises(RuntimeError, match="approved"):
        store.build_dataset(tok, version="v1")


def test_dataset_build_split_and_provenance(tmp_path, corpus):
    store = ScratchDatasetStore(str(tmp_path / "store"))
    # ingest is idempotent: identical text is not double-counted
    examples = [store.ingest_text(t, source="synthetic") for t in corpus]
    assert len(examples) == 20
    store.approve_all()
    approved = store.list_approved()
    assert len(approved) == 4  # 4 unique texts in the synthetic corpus
    tok = ScratchTokenizer(vocab_size=300)
    tok.train([e.text for e in approved])
    manifest = store.build_dataset(tok, version="v1", val_fraction=0.25, seed=7)
    assert manifest["format"] == "soup-scratch-dataset"
    assert manifest["split"]["train_examples"] + manifest["split"]["val_examples"] == len(approved)
    assert manifest["provenance"]["sources"] == ["synthetic"]
    assert manifest["provenance"]["approved_only"] is True

    loaded = store.load_dataset("v1")
    assert loaded["train_ids"].dtype == np.int32
    assert len(loaded["train_ids"]) == manifest["split"]["train_tokens"]
    with open(os.path.join(store.root, "datasets", "v1", "manifest.json")) as fh:
        on_disk = json.load(fh)
    assert on_disk["dataset_sha256"] == manifest["dataset_sha256"]


def test_duplicate_ingest_is_idempotent(tmp_path):
    store = ScratchDatasetStore(str(tmp_path / "store"))
    ex1 = store.ingest_text("duplicate text", source="a")
    ex2 = store.ingest_text("duplicate text", source="b")
    assert ex1.example_id == ex2.example_id
    assert len(store.list_pending()) == 1


def test_dataset_version_conflict(tmp_path, corpus):
    store = ScratchDatasetStore(str(tmp_path / "store"))
    for t in corpus:
        store.ingest_text(t)
    store.approve_all()
    tok = ScratchTokenizer(vocab_size=300)
    tok.train([e.text for e in store.list_approved()])
    store.build_dataset(tok, version="v1")
    with pytest.raises(ValueError, match="already exists"):
        store.build_dataset(tok, version="v1")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def test_model_config_validation():
    with pytest.raises(ValueError):
        ScratchModelConfig(d_model=10, n_heads=4)  # not divisible
    with pytest.raises(ValueError):
        ScratchModelConfig(n_layers=0)


def test_model_forward_and_loss():
    torch = pytest.importorskip("torch")
    cfg = ScratchModelConfig(vocab_size=280, block_size=16, n_layers=2, n_heads=2, d_model=16)
    model = ScratchTransformer(cfg).build()
    x = torch.randint(0, 280, (2, 16))
    y = torch.randint(0, 280, (2, 16))
    loss, logits = model(x, y)
    assert logits.shape == (2, 16, 280)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_model_backward_and_step_changes_loss():
    torch = pytest.importorskip("torch")
    cfg = ScratchModelConfig(vocab_size=280, block_size=16, n_layers=2, n_heads=2, d_model=16)
    model = ScratchTransformer(cfg).build()
    torch.manual_seed(0)
    x = torch.randint(0, 280, (4, 16))
    y = torch.randint(0, 280, (4, 16))
    loss1, _ = model(x, y)
    loss1.backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.step()
    loss2, _ = model(x, y)
    assert loss2.item() < loss1.item()  # the model actually learns


def test_model_sequence_exceeds_block_size():
    torch = pytest.importorskip("torch")
    cfg = ScratchModelConfig(vocab_size=280, block_size=8, n_layers=1, n_heads=2, d_model=16)
    model = ScratchTransformer(cfg).build()
    x = torch.randint(0, 280, (1, 16))
    with pytest.raises(ValueError, match="block_size"):
        model(x)


def test_model_is_randomly_initialised_not_pretrained():
    torch = pytest.importorskip("torch")
    cfg = ScratchModelConfig(
        vocab_size=280, block_size=8, n_layers=1, n_heads=2, d_model=16, seed=123
    )
    m1 = ScratchTransformer(cfg).build()
    m2 = ScratchTransformer(cfg).build()
    # seeded: identical inits; but no zero/one pretrained patterns
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
        assert not torch.allclose(p1, torch.zeros_like(p1))


# --------------------------------------------------------------------------- #
# training engine
# --------------------------------------------------------------------------- #


def _tiny_setup(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    train_ids = rng.integers(0, 280, size=2048).astype(np.int32)
    val_ids = rng.integers(0, 280, size=512).astype(np.int32)
    cfg = ScratchModelConfig(vocab_size=280, block_size=16, n_layers=1, n_heads=2, d_model=16)
    model = ScratchTransformer(cfg).build()
    tcfg = ScratchTrainingConfig(
        epochs=1,
        batch_size=4,
        block_size=16,
        eval_interval=2,
        save_interval=2,
        log_interval=1,
        max_steps=4,
        eval_batches=2,
    )
    trainer = ScratchTrainer(
        model=model,
        train_ids=train_ids,
        val_ids=val_ids,
        config=tcfg,
        checkpoint_dir=str(tmp_path / "ckpt"),
        model_config=cfg.to_dict(),
    )
    return trainer, model, tcfg


def test_training_loop_runs_and_checkpoints(tmp_path):
    trainer, model, tcfg = _tiny_setup(tmp_path)
    result = trainer.train()
    assert result.steps == tcfg.max_steps
    assert result.checkpoint_dir
    state_file = os.path.join(result.checkpoint_dir, "training_state.json")
    assert os.path.isfile(state_file)
    with open(state_file) as fh:
        state = json.load(fh)
    assert state["step"] == result.steps
    assert os.path.isfile(os.path.join(result.checkpoint_dir, "loss_log.jsonl"))
    assert os.path.isfile(os.path.join(result.checkpoint_dir, "optimizer.pt"))
    # model tensors persisted
    assert os.path.isfile(os.path.join(result.checkpoint_dir, "model.safetensors"))


def test_resume_from_checkpoint(tmp_path):
    trainer, model, tcfg = _tiny_setup(tmp_path)
    result = trainer.train()
    trained_step = result.steps

    # resume: restore into a fresh instance and confirm state round-trips
    cfg = ScratchModelConfig(vocab_size=280, block_size=16, n_layers=1, n_heads=2, d_model=16)
    model2 = ScratchTransformer(cfg).build()
    trainer2 = ScratchTrainer(
        model=model2,
        train_ids=trainer.train_ids,
        val_ids=trainer.val_ids,
        config=tcfg,
        checkpoint_dir=result.checkpoint_dir,
        model_config=cfg.to_dict(),
    )
    torch = pytest.importorskip("torch")
    trainer2.checkpoint.restore_model(model2)
    state = trainer2.checkpoint.load()
    assert state["step"] == trained_step
    # restored weights match the saved model
    saved = ScratchTransformer(cfg).build()
    trainer2.checkpoint.restore_model(saved)
    for p1, p2 in zip(model2.parameters(), saved.parameters()):
        assert torch.allclose(p1, p2)


def test_batch_too_short_raises(tmp_path):
    tcfg = ScratchTrainingConfig(block_size=64, max_steps=1)
    pytest.importorskip("torch")
    cfg = ScratchModelConfig(vocab_size=280, block_size=64, n_layers=1, n_heads=2, d_model=16)
    model = ScratchTransformer(cfg).build()
    trainer = ScratchTrainer(
        model=model,
        train_ids=np.arange(10, dtype=np.int32),
        val_ids=np.arange(10, dtype=np.int32),
        config=tcfg,
        checkpoint_dir=str(tmp_path / "c"),
    )
    with pytest.raises(ValueError, match="too short"):
        trainer._get_batch("train", 2)


def test_evaluation_real_metrics(tmp_path):
    trainer, model, tcfg = _tiny_setup(tmp_path)
    trainer.train()
    loss = trainer.evaluate()
    assert 0.0 < loss < 50.0
    report = evaluate_model(
        model=model, val_ids=trainer.val_ids, block_size=tcfg.block_size, eval_batches=2
    )
    assert report["val_perplexity"] >= 1.0
    stats = dataset_statistics({"version": "v1", "split": {"train_tokens": 10}, "tokenizer": {}})
    assert stats["train_tokens"] == 10


# --------------------------------------------------------------------------- #
# remote training (never local)
# --------------------------------------------------------------------------- #


def test_remote_payload_bundle_roundtrip(tmp_path, corpus):
    from soup_cli.scratch.remote import build_payload_from_store

    store = ScratchDatasetStore(str(tmp_path / "store"))
    for t in corpus:
        store.ingest_text(t)
    store.approve_all()
    tok = ScratchTokenizer(vocab_size=300)
    tok.train([e.text for e in store.list_approved()])
    store.build_dataset(tok, version="v1")
    payload = build_payload_from_store(
        store,
        dataset_version="v1",
        model_config=ScratchModelConfig().to_dict(),
        training_config=ScratchTrainingConfig().to_dict(),
    )
    bundle = payload.to_bundle()
    decoded = json.loads(__import__("base64").b64decode(bundle))
    assert decoded["format"] == "soup-scratch-remote"
    assert decoded["model_config"]["architecture"] == "causal-transformer"


def test_remote_plan_only_renders_stub(tmp_path, corpus):
    from soup_cli.cloud._common import CloudPlan
    from soup_cli.scratch.learn import LearnWorkflow

    wf = LearnWorkflow(str(tmp_path / "learn"))
    for t in corpus:
        wf.ingest_text(t)
    wf.approve(all_pending=True)
    wf.prepare_dataset(version="v1", vocab_size=300)
    plan = wf.plan_remote_training(
        dataset_version="v1",
        model_config=ScratchModelConfig(block_size=16, d_model=16, n_layers=1).to_dict(),
        training_config=ScratchTrainingConfig(block_size=16, max_steps=2).to_dict(),
        gpu="t4",
        output_dir=str(tmp_path / "out"),
    )
    assert isinstance(plan, CloudPlan)
    assert plan.cloud == "modal"
    assert "modal run" in plan.run_command
    assert "soup-scratch-remote" in plan.stub_text


def test_remote_submit_without_credentials_fails_loudly(tmp_path, corpus):
    from soup_cli.scratch import remote as scratch_remote
    from soup_cli.scratch.learn import LearnWorkflow

    wf = LearnWorkflow(str(tmp_path / "learn"))
    for t in corpus:
        wf.ingest_text(t)
    wf.approve(all_pending=True)
    wf.prepare_dataset(version="v1", vocab_size=300)
    plan = wf.plan_remote_training(
        dataset_version="v1",
        model_config=ScratchModelConfig(block_size=16, d_model=16, n_layers=1).to_dict(),
        training_config=ScratchTrainingConfig(block_size=16, max_steps=2).to_dict(),
        gpu="t4",
        output_dir=str(tmp_path / "out"),
    )
    # no Modal token, no ~/.modal.toml, no modal SDK — must RAISE, never fall back
    with pytest.raises((RuntimeError, Exception)):
        scratch_remote.submit_scratch_modal_run(plan, env={})


def test_invalid_gpu_refused():
    from soup_cli.scratch.remote import render_scratch_modal_stub

    with pytest.raises(ValueError):
        render_scratch_modal_stub("x", gpu="not-a-gpu", output_dir="out", soup_version="0.0.0")


# --------------------------------------------------------------------------- #
# learn workflow integration + CLI
# --------------------------------------------------------------------------- #


def test_learn_register_requires_remote_evidence(tmp_path, corpus):
    from soup_cli.scratch.learn import LearnWorkflow

    wf = LearnWorkflow(str(tmp_path / "learn"))
    for t in corpus:
        wf.ingest_text(t)
    wf.approve(all_pending=True)
    wf.prepare_dataset(version="v1", vocab_size=300)
    fake_run = tmp_path / "fake_run"
    (fake_run / "checkpoint-final").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="run_report"):
        wf.register_run(run_dir=str(fake_run), dataset_version="v1", name="should-fail")


def test_learn_state_records_plans(tmp_path, corpus):
    from soup_cli.scratch.learn import LearnWorkflow

    wf = LearnWorkflow(str(tmp_path / "learn"))
    for t in corpus:
        wf.ingest_text(t)
    wf.approve(all_pending=True)
    wf.prepare_dataset(version="v1", vocab_size=300)
    wf.plan_remote_training(
        dataset_version="v1",
        model_config=ScratchModelConfig(block_size=16, d_model=16, n_layers=1).to_dict(),
        training_config=ScratchTrainingConfig(block_size=16, max_steps=2).to_dict(),
        gpu="t4",
        output_dir=str(tmp_path / "out"),
    )
    with open(wf.paths.learn_state) as fh:
        state = json.load(fh)
    assert "v1" in state["plans"]
    assert state["plans"]["v1"]["gpu"] == "t4"


def test_cli_registered():

    from soup_cli.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["learn", "--help"])
    assert result.exit_code == 0
    for sub in (
        "ingest",
        "review",
        "approve",
        "prepare",
        "train",
        "evaluate",
        "register",
        "status",
    ):
        assert sub in result.output


def test_cli_train_is_plan_only_by_default(tmp_path, corpus):
    import os as _os
    import tempfile

    from soup_cli.cli import app

    runner = CliRunner()
    old_cwd = _os.getcwd()
    workdir = tempfile.mkdtemp(dir=str(tmp_path))
    _os.chdir(workdir)
    try:
        for t in corpus:
            r = runner.invoke(app, ["learn", "ingest", "--text", t, "--root", "learn"])
            assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["learn", "approve", "--all", "--root", "learn"])
        assert r.exit_code == 0, r.output
        r = runner.invoke(
            app, ["learn", "prepare", "--version", "v1", "--vocab-size", "300", "--root", "learn"]
        )
        assert r.exit_code == 0, r.output
        result = runner.invoke(
            app,
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
                "--root",
                "learn",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "plan-only" in result.output.lower()
    finally:
        _os.chdir(old_cwd)

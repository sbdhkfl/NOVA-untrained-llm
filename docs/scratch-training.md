# From-Scratch Training (`soup learn`)

SOUP's from-scratch training subsystem trains a **genuinely untrained
language model** — no pretrained weights, no HF checkpoints — progressively
from approved learning material. This is the infrastructure behind:

```bash
soup learn <request>      ->  pending examples (free-form: topic / document / dataset)
        (or: soup learn ingest --text "..."  |  soup learn start --from F)
soup learn review  ->  soup learn approve  ->  soup learn edit (optional)
                   ->  soup learn prepare ->  soup learn train   (REMOTE, Modal)
                   ->  soup learn evaluate -> soup learn register
                   ->  soup learn generate / soup learn models / soup learn status
```

Everything lives in `src/soup_cli/scratch/`.

## Model architecture

`src/soup_cli/scratch/model.py` + `_torch_model.py` — a decoder-only causal
transformer built from scratch:

| Component | Implementation |
|---|---|
| Token embedding | `nn.Embedding(vocab_size, d_model)` |
| Positional encoding | learned positional embeddings (`block_size` rows) |
| Attention | causal multi-head SDPA (`F.scaled_dot_product_attention`, `is_causal=True`) |
| Feed-forward | GELU / ReLU / SwiGLU variants, configurable multiplier |
| Normalization | pre-norm RMSNorm or LayerNorm |
| Output head | linear LM head, optionally weight-tied to the embedding |
| Loss | cross-entropy over the shifted token stream (`ignore_index=-1`) |
| Init | seeded `normal(0, 0.02)` / zero-bias GPT-2-style init — **there is no pretrained path** |

Configuration dataclass: `ScratchModelConfig` (`vocab_size`, `block_size`,
`n_layers`, `n_heads`, `d_model`, `ffn_multiplier`, `dropout`, `norm_type`,
`activation`, `tie_weights`, `seed`). `param_count()` estimates parameters
without instantiating torch.

## Tokenizer

`src/soup_cli/scratch/tokenizer.py` — byte-level BPE trained from the
approved learning material itself:

- **Lossless round-trip**: byte-level base guarantees any UTF-8 text
  encodes and decodes identically.
- **Special tokens**: `<pad>`(256), `<bos>`(257), `<eos>`(258), `<unk>`(259),
  plus user extras (`add_special_tokens` before training).
- **Padding**: `pad()` / `pad_batch()` with the `<pad>` id.
- **Versioning**: `version_hash` = SHA-256 over (vocab_size, special tokens,
  merges). Saved in every dataset manifest and checkpoint.
- **Persistence**: JSON (`format: soup-scratch-bpe`) via `save()` / `load()`;
  incompatible files are refused.
- Training and inference use the same `encode()`/`decode()`.

## Dataset pipeline

`src/soup_cli/scratch/dataset.py` — on-disk store with a **hard approval
gate**:

```
<root>/pending/<id>.json      reviewable, NOT approved
<root>/approved/<id>.json     safe to train on
<root>/rejected/<id>.json     never trained on
<root>/datasets/<version>/    tokenised .npy + manifest.json + tokenizer.json
```

Stages: `ingest` (clean → NFC, control-char strip, whitespace collapse;
validate → min/max chars, printable ratio; dedupe → idempotent ingest by
SHA-256, explicit `dedupe()`) → `review` → `approve` (**explicit only**) →
`build_dataset` (tokenize with bos/eos framing, deterministic seeded
train/val split, versioned manifest with provenance).

`build_dataset` **refuses to run with zero approved examples**. Manifest
fields: `version`, `dataset_sha256`, `provenance` (example ids, sources,
seed, `approved_only`), `split` (token/example counts), `tokenizer` hash.

Dataset format: `train_ids.npy` / `val_ids.npy` are flat int32 token
streams.

## Training configuration

`src/soup_cli/scratch/engine.py` — `ScratchTrainingConfig`:

| Field | Meaning |
|---|---|
| `epochs`, `batch_size`, `block_size` | loop shape (must match model `block_size`) |
| `learning_rate`, `optimizer` (`adamw`/`sgd`), `weight_decay` | optimization |
| `grad_clip` | global-norm gradient clipping |
| `warmup_steps`, `lr_scheduler` (`cosine`/`constant`) | schedule |
| `eval_interval`, `eval_batches`, `save_interval`, `log_interval` | cadence |
| `max_steps` | cap (0 = full run) |
| `seed`, `device` | reproducibility / device |

`ScratchTrainer.train(resume=...)` runs the real loop: forward → CE loss →
backward → clip → optimizer step, with loss tracking (`loss_log.jsonl`),
periodic validation, and checkpointing. Resume restores model weights,
optimizer state, step counter and RNG state.

## Remote compute — training NEVER runs locally

`src/soup_cli/scratch/remote.py` + `remote_train.py` reuse SOUP's Modal
infrastructure (`soup_cli.cloud.modal`):

1. `build_payload_from_store` packs model config, training config,
   tokenizer, and the tokenised splits into one base64 bundle
   (data only — no code interpolation, no secrets).
2. `plan_scratch_modal_run` renders a self-contained Modal stub: image with
   `soup-cli[train]` + torch, GPU from the allowlist
   (`t4/l4/a10g/a100/a100-80gb/l40s/h100`), 4h timeout, outputs on the
   `soup-outputs` volume under the run name.
3. Default is **plan-only**; `--cloud-submit` performs the live submit
   (Modal token required).
4. All output files are downloaded back to `output_dir` when the run ends
   (also after failure).

**Guarantees**: the local machine never downloads weights for training,
never executes training workloads, never uses a local GPU. There is **no
silent fallback to local training**: missing Modal credentials, missing SDK,
or missing `modal` on PATH all raise a clear `RuntimeError`. The worker
(`remote_train.py`) runs only inside the Modal container and forces
`device="cuda"`.

The learn state file (`learn/learn_state.json`) records every plan
(stub path, run command, GPU, dataset/model/training config) for
reproducibility.

## Checkpoint format

A checkpoint directory contains:

| File | Contents |
|---|---|
| `model.safetensors` | model tensors (safetensors; torch fallback; tied weights deduplicated) |
| `optimizer.pt` | optimizer state (resume) |
| `training_state.json` | step, epoch, RNG state, training + model config |
| `tokenizer.json` | the tokenizer snapshot (compatibility check) |
| `loss_log.jsonl` | per-interval train/val loss, lr, grad norm, throughput |

Remote runs additionally write `run_report.json` (final losses, perplexity,
config hashes) — **registration requires this file**, so a run can only be
registered with real remote-training evidence.

## Evaluation

`src/soup_cli/scratch/eval.py` computes, from real forward passes:

- validation loss (cross-entropy),
- **perplexity** `exp(val_loss)`,
- dataset statistics (tokens, examples, vocab size, dataset hash),
- checkpoint provenance (step, run dir).

Reports are written as JSON (`eval_report.json` next to the checkpoint).
No evaluation number is ever fabricated.

## Registry integration

`soup learn register` pushes to `RegistryStore` with
`base_model=scratch:causal-transformer`, `task=scratch`, the full
model/training/dataset config in `config`, metrics, dataset-manifest
provenance (`data_path`), and the checkpoint directory as an artifact.

## Free-form learning requests

`soup learn` accepts a plain-language request; an unknown first word routes
through the `start` command (mirroring `soup bench`'s router):

```bash
soup learn Python                       # topic: needs learn/material/<slug>/ or --from
soup learn how computers work           # multi-word topics work as-is
soup learn mathematics --from notes.md  # topic with an explicit material file
soup learn from this document report.md # ingest a document's text
soup learn from this dataset data.jsonl # ingest dataset rows (alpaca/chat/text)
```

Every path lands examples in `pending` — nothing is trained on until you
explicitly approve. Material for a topic is looked up at
`learn/material/<topic-slug>/` (files ingested in sorted order) when `--from`
is not given; a topic with no material exits 2 with guidance instead of
silently doing nothing.

The rest of the review loop:

```bash
soup learn edit <id> --text "..."   # fix an example; it returns to pending
soup learn status                   # examples, dataset versions, plan progress, models
soup learn generate <run> --prompt "..." # inference from a run's checkpoint
soup learn models                   # registered from-scratch models
```

## Training workflow (end to end)

```bash
# 1. provide learning material (lands in pending — nothing is trained on yet)
soup learn ingest --text "..."          # or: soup learn ingest ./corpus.txt
# 2. review
soup learn review
# 3. explicitly approve
soup learn approve --all                # or: soup learn approve <id> [<id> ...]
soup learn reject <id>                  # or reject individual examples
# 4. build the versioned tokenised dataset (fails without approvals)
soup learn prepare --version v1 --vocab-size 512
# 5. plan remote training (default) ...
soup learn train --dataset v1 --gpu a10g --epochs 3
#    ... or submit for real
soup learn train --dataset v1 --gpu a10g --cloud-submit
# 6. evaluate the downloaded checkpoint (run dir OR checkpoint dir works)
soup learn evaluate learn/runs/v1-run --dataset v1
# 7. register
soup learn register learn/runs/v1-run --dataset v1 --name my-scratch-model
# 8. inference / bookkeeping
soup learn generate learn/runs/v1-run --prompt "def add(" --max-tokens 64
soup learn models
soup learn status
```

## Tests

- `tests/scratch/test_scratch_training.py` — 30 tests on small synthetic
  data: tokenizer round-trip/special tokens/padding/persistence, dataset
  validation/approval gate/splits/dedupe/provenance, model init/forward/loss/
  backward/learning-signal, training loop/checkpoint/resume, evaluation,
  remote payload/plan/failed-submit (loud failure, never local fallback),
  CLI wiring.
- `tests/scratch/test_learn_integration.py` — 24 integration tests over the
  whole mission path: request parsing, topic/document/dataset submission,
  edit/approve/prepare failure handling, stub writing (volume mount + payload
  decode regression guards), training-job submission through the shared
  Modal override seam (and loud submit failure), status retrieval, and the
  full evaluate → register → models → generate workflow. Local only: tiny
  synthetic data, mocked remote submission, no downloads.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no approved examples: refusing to build a dataset` | Approve examples first (`soup learn approve`). This is the safety gate, not a bug. |
| `token stream too short` | Increase corpus size or lower `block_size`; the stream must exceed `block_size + 1` tokens. |
| `Modal not authenticated` | Run `modal setup` or set `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`; then `--cloud-submit`. Never falls back to local. |
| `Modal SDK not installed` / `modal not on PATH` | `pip install "soup-cli[modal]"`. |
| `payload exceeds … bytes` | Corpus too large for inline embedding; split into versions or use the volume-based flow. |
| `not a soup-scratch BPE tokenizer` | Checkpoint/dataset tokenizer mismatch; re-prepare the dataset with the matching tokenizer. |
| `dataset version … already exists` | Use a new version tag; datasets are immutable by design. |
| `no run_report.json` at registration | The checkpoint was not produced by a real remote run; registration is refused without remote evidence. |
| Stub image build fails with `Could not find a version that satisfies the requirement soup-cli==…` | The Modal image pip-installs the pinned `soup-cli` version from PyPI; until this release is published, run training after the release ships, or change the pin in the generated `soup_learn_modal_app.py` to a published version that carries `soup_cli.scratch`. |
| `no learning material for …` (exit 2) | A topic needs material: put files under `learn/material/<topic-slug>/` or pass `--from <file-or-dir>`. |
| Loss stuck at `ln(vocab_size)` ≈ uniform | Model/step budget too small, LR too low, or data has no learnable structure — increase `max_steps`, `d_model`, or LR. |

## Coordination notes

For OpenCode (integration):

- CLI: `app.add_typer(learn.app, name="learn")` in `src/soup_cli/cli.py`
  (subcommands: `start`, `ingest`, `review`, `approve`, `reject`, `edit`,
  `prepare`, `train`, `evaluate`, `register`, `generate`, `models`,
  `status`; free-form `soup learn <request>` routes to `start` via the
  `_LearnGroup` TyperGroup subclass, mirroring `_BenchGroup`).
- No schema changes were made to `src/soup_cli/config/schema.py` — the
  from-scratch configs are self-contained dataclasses
  (`ScratchModelConfig`, `ScratchTrainingConfig`). If you prefer routing
  through `SoupConfig`, add a `scratch` section and mirror the fields.
- Heavy deps (torch, numpy) are lazy-imported inside functions everywhere;
  the package imports cleanly without torch.
- Registry push uses the existing `RegistryStore.push` /
  `add_artifact` contract; `add_artifact` is keyword-only
  (`entry_id=`, `kind=`, `path=`) and `"checkpoint"` was added to
  `_VALID_KINDS`.
- Modal integration reuses `CloudPlan`, `write_cloud_stub`, and the
  `submit_modal_run` auth gating (`_MODAL_SUBMIT_OVERRIDE` seam is what
  tests monkeypatch); the scratch layer only adds its own stub template
  and payload builder.

For Deep AI Code (upstream author):

- Integration review found and fixed upstream: missing volume mount +
  `volume.commit()` in the stub template (checkpoints would have been
  lost), unverified `batch_download` replaced with `listdir`/`read_file`
  download, `plan_remote_training` never wrote the stub file,
  `add_artifact` called positionally, checkpoint artifact path pointing at
  a directory, model `vocab_size` not synced to the dataset tokenizer
  (remote embedding index-out-of-range), BOM not stripped by
  `clean_text`, and missing friendly CLI error handling.
- New since your commit: `start` (free-form router), `edit`, `generate`,
  `models`, rewritten `status`, `resolve_checkpoint_dir` (run dir OR
  checkpoint dir), `inference.py`, and `tests/scratch/test_learn_integration.py`.
- Please keep remote-only guarantees when extending: plan-only default,
  loud failures, no local fallback, no large local downloads.

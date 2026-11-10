# From-Scratch Training (`soup learn`)

SOUP's from-scratch training subsystem trains a **genuinely untrained
language model** — no pretrained weights, no HF checkpoints — progressively
from approved learning material. This is the infrastructure behind:

```bash
soup learn ingest  ->  soup learn review  ->  soup learn approve
                   ->  soup learn prepare ->  soup learn train   (REMOTE, Modal)
                   ->  soup learn evaluate -> soup learn register
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
# 6. evaluate the downloaded checkpoint
soup learn evaluate runs/v1-<run>/checkpoint-final --dataset v1
# 7. register
soup learn register runs/v1-<run> --dataset v1 --name my-scratch-model
```

## Tests

`tests/scratch/test_scratch_training.py` — 30 tests, all on small synthetic
data: tokenizer round-trip/special tokens/padding/persistence, dataset
validation/approval gate/splits/dedupe/provenance, model init/forward/loss/
backward/learning-signal, training loop/checkpoint/resume, evaluation,
remote payload/plan/failed-submit (loud failure, never local fallback),
CLI wiring.

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
| Loss stuck at `ln(vocab_size)` ≈ uniform | Model/step budget too small, LR too low, or data has no learnable structure — increase `max_steps`, `d_model`, or LR. |

## Coordination notes for OpenCode (integration)

- CLI: `app.add_typer(learn.app, name="learn")` in `src/soup_cli/cli.py`
  (subcommands: `ingest`, `review`, `approve`, `reject`, `prepare`,
  `train`, `evaluate`, `register`, `status`).
- No schema changes were made to `src/soup_cli/config/schema.py` — the
  from-scratch configs are self-contained dataclasses
  (`ScratchModelConfig`, `ScratchTrainingConfig`). If you prefer routing
  through `SoupConfig`, add a `scratch` section and mirror the fields.
- Heavy deps (torch, numpy) are lazy-imported inside functions everywhere;
  the package imports cleanly without torch.
- Registry push uses the existing `RegistryStore.push` /
  `add_artifact` contract; no registry changes required.
- Modal integration reuses `CloudPlan`, `write_cloud_stub`, and the
  `submit_modal_run` auth gating; the scratch layer only adds its own stub
  template and payload builder.

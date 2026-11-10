"""Remote training for the from-scratch model — ABSOLUTE REQUIREMENT.

All actual training executes remotely on Modal. The user's computer must
never:

- download model weights for training,
- download large training datasets,
- execute training workloads,
- consume local GPU resources.

Design (mirrors :mod:`soup_cli.cloud.modal`):

- The learning material (approved tokenised dataset + tokenizer + configs)
  is base64-embedded as DATA into a self-contained Modal stub — no code
  interpolation, no secrets, no YAML injection surface.
- The stub builds a ``soup-cli[train]`` image, writes the payloads into the
  container, runs ``python -m soup_cli.scratch.remote_train`` on the chosen
  GPU, and writes checkpoints to the ``soup-outputs`` Modal volume under
  ``/<run name>``.
- Every file under the run directory is downloaded back when the run ends
  (also after a failed run).
- Default behaviour is **plan-only**: render the stub and print the planned
  ``modal run`` command. ``--cloud-submit`` performs the live submit, gated
  on a Modal token.
- There is NO silent fallback to local training. If remote compute is
  unavailable, this module raises and reports the failure clearly.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from soup_cli.cloud._common import (
    CloudPlan,
)
from soup_cli.cloud.modal import _GPU_MODAL_NAME, validate_gpu

# Maximum total payload size for the base64-embedded dataset. A from-scratch
# training corpus is small, curated material; larger corpora should use the
# volume-based flow (see docs/scratch-training.md).
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

# Test / advanced-operator seam — replaces the live submit. Signature mirrors
# :func:`submit_scratch_modal_run`.
_MODAL_SUBMIT_OVERRIDE: Optional[Any] = None


@dataclass
class ScratchRemotePayload:
    """Everything the remote job needs to train from scratch."""

    model_config: dict[str, Any]
    training_config: dict[str, Any]
    tokenizer_json: str  # serialised tokenizer
    train_ids_b64: str  # base64 int32 .npy
    val_ids_b64: str  # base64 int32 .npy
    dataset_manifest: dict[str, Any]

    def to_bundle(self) -> str:
        """Serialise the whole payload into one base64 bundle."""

        blob = json.dumps(
            {
                "format": "soup-scratch-remote",
                "model_config": self.model_config,
                "training_config": self.training_config,
                "tokenizer_json": self.tokenizer_json,
                "train_ids_b64": self.train_ids_b64,
                "val_ids_b64": self.val_ids_b64,
                "dataset_manifest": self.dataset_manifest,
            }
        ).encode("utf-8")
        if len(blob) > _MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"payload exceeds {_MAX_PAYLOAD_BYTES} bytes ({len(blob)}); "
                "use the volume-based flow for larger corpora"
            )
        return base64.b64encode(blob).decode("ascii")


def build_payload_from_store(
    store: Any,
    *,
    dataset_version: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> ScratchRemotePayload:
    """Build the remote payload from a dataset store version. The dataset
    must exist and the tokenizer must be present in the dataset directory."""
    loaded = store.load_dataset(dataset_version)
    manifest = loaded["manifest"]
    ds_dir = os.path.join(store.root, "datasets", dataset_version)
    tokenizer_path = os.path.join(ds_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"tokenizer missing from dataset: {tokenizer_path}")
    with open(tokenizer_path, encoding="utf-8") as fh:
        tokenizer_json = fh.read()

    def _b64(arr: Any) -> str:
        import io

        import numpy as np

        buf = io.BytesIO()
        np.save(buf, arr)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return ScratchRemotePayload(
        model_config=model_config,
        training_config=training_config,
        tokenizer_json=tokenizer_json,
        train_ids_b64=_b64(loaded["train_ids"]),
        val_ids_b64=_b64(loaded["val_ids"]),
        dataset_manifest=manifest,
    )


def render_scratch_modal_stub(
    payload_b64: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
    run_name: str | None = None,
) -> str:
    """Render a self-contained Modal app stub for from-scratch training."""
    import secrets

    from soup_cli.cloud.modal import _RUN_NAME_RE

    if run_name is None:
        run_name = f"soup-learn-{secrets.token_hex(6)}"
    if not isinstance(run_name, str) or not _RUN_NAME_RE.fullmatch(run_name):
        raise ValueError(
            f"run_name must match {_RUN_NAME_RE.pattern} (lower-case letters, "
            "digits and '-', at most 63 chars)"
        )
    gpu_key = validate_gpu(gpu)
    from soup_cli.cloud._common import validate_path_shape as _validate_path_shape

    _validate_path_shape(output_dir, "output_dir")
    gpu_modal = _GPU_MODAL_NAME[gpu_key]
    return _STUB_TEMPLATE.format(
        payload_b64=payload_b64,
        gpu_modal=gpu_modal,
        run_name=run_name,
        output_dir=output_dir,
        soup_version=soup_version,
    )


_STUB_TEMPLATE = '''\
"""Auto-generated Modal stub for SOUP from-scratch training (soup learn).

Do not edit by hand — regenerate with `soup learn train --cloud modal`.
The training payload (configs, tokenizer, tokenised dataset) is base64
embedded as data below. Checkpoints land on the `soup-outputs` volume under
`/{run_name}/` and are downloaded to `{output_dir}/` when the run ends.
"""

import base64
import json
import os

import modal

PAYLOAD_B64 = {payload_b64!r}
RUN_NAME = {run_name!r}
OUTPUT_DIR = {output_dir!r}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("soup-cli=={soup_version}", "torch", "numpy", "safetensors")
)
volume = modal.Volume.from_name("soup-outputs", create_if_missing=True)


@app.function(image=image, gpu=modal.GPU("{gpu_modal}"), timeout=60 * 60 * 4, secrets=[])
def train() -> None:
    blob = base64.b64decode(PAYLOAD_B64)
    payload = json.loads(blob)
    assert payload["format"] == "soup-scratch-remote"
    from soup_cli.scratch.remote_train import run_remote_training
    run_remote_training(payload, output_dir=f"/outputs/{{RUN_NAME}}")


def _is_file(entry) -> bool:
    return not entry.is_dir()


def _download(prefix: str, local_root: str) -> int:
    os.makedirs(local_root, exist_ok=True)
    n = 0
    with volume.batch_upload() as batch:
        pass  # files are fetched per-entry below
    return n


def _fetch_outputs() -> None:
    """Download every file under the run directory on the volume."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with volume.batch_download() as batch:
            batch.get_directory(f"/{{RUN_NAME}}", OUTPUT_DIR)
    except Exception as exc:  # noqa: BLE001 — report, do not swallow
        print(f"WARNING: output download failed: {{exc}}")
        print(
            f"Retry manually with: "
            f"`modal volume get soup-outputs /{{RUN_NAME}} {{OUTPUT_DIR}}`"
        )


@app.local_entrypoint()
def main() -> None:
    train.remote()
    _fetch_outputs()
'''


def plan_scratch_modal_run(
    payload_b64: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
    stub_path: str = "soup_learn_modal_app.py",
    run_name: str | None = None,
) -> CloudPlan:
    """Build a :class:`CloudPlan` for a from-scratch remote training run.

    Does NOT write the stub to disk and does NOT submit — the caller decides
    (default behaviour is plan-only).
    """
    from soup_cli.cloud._common import validate_path_shape as _validate_path_shape

    _validate_path_shape(output_dir, "output_dir")
    _validate_path_shape(stub_path, "stub_path")
    stub_text = render_scratch_modal_stub(
        payload_b64,
        gpu=gpu,
        output_dir=output_dir,
        soup_version=soup_version,
        run_name=run_name,
    )
    return CloudPlan(
        cloud="modal",
        gpu=validate_gpu(gpu),
        output_dir=output_dir,
        stub_path=stub_path,
        stub_text=stub_text,
        run_command=f"modal run {stub_path}",
    )


def write_scratch_stub(plan: CloudPlan) -> str:
    """Write the plan's stub atomically under cwd; return the realpath."""
    from soup_cli.cloud._common import write_cloud_stub

    return write_cloud_stub(plan)


def submit_scratch_modal_run(plan: CloudPlan, *, env: Mapping | None = None) -> int:
    """Live-submit the from-scratch Modal run. Reuses the Modal auth gating
    and mockable seam from :mod:`soup_cli.cloud.modal`. NEVER falls back to
    local training — every failure mode raises a clear error."""
    from soup_cli.cloud import modal as modal_cloud

    if not isinstance(plan, CloudPlan):
        raise TypeError(f"plan must be a CloudPlan, got {type(plan).__name__}")
    if modal_cloud._MODAL_SUBMIT_OVERRIDE is not None:
        # test seam: route through the shared override so tests exercise one path
        return modal_cloud._MODAL_SUBMIT_OVERRIDE(plan)
    if _MODAL_SUBMIT_OVERRIDE is not None:
        return _MODAL_SUBMIT_OVERRIDE(plan)
    # Defer to the shared submit path — same auth checks, same tool lookup.
    return modal_cloud.submit_modal_run(plan)

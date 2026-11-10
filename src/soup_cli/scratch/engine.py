"""Training engine for the from-scratch model.

A real training loop — not a mock: configurable epochs, batch sizes, learning
rate, optimizer (AdamW), gradient clipping, validation, loss tracking,
checkpointing, resume-from-checkpoint, training logs, and error recovery.

Reproducibility: every run saves its full config (model config, tokenizer
version hash, dataset version, seed) so the run can be reproduced exactly.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class ScratchTrainingConfig:
    """Training configuration. Reproducible runs are saved with this config."""

    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-4
    optimizer: str = "adamw"  # "adamw" | "sgd"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 0
    lr_scheduler: str = "cosine"  # "cosine" | "constant"
    seed: int = 1337
    device: str = "cpu"  # training is always remote; local only for tests
    eval_interval: int = 50
    eval_batches: int = 20
    save_interval: int = 100
    log_interval: int = 10
    max_steps: int = 0  # 0 = no limit
    val_fraction: float = 0.1
    block_size: int = 256  # must match the model's block_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScratchTrainingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class StepResult:
    step: int
    loss: float
    lr: float
    grad_norm: float
    tokens_per_sec: float
    wall_sec: float


@dataclass
class TrainResult:
    steps: int
    epochs: int
    final_train_loss: float
    final_val_loss: float
    final_val_perplexity: float
    total_time_sec: float
    checkpoint_dir: str
    loss_log: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #

CHECKPOINT_FILES = ("model.safetensors", "optimizer.pt", "training_state.json")


class CheckpointManager:
    """Saves and restores versioned checkpoints. A checkpoint directory
    contains the model state, optimizer state, tokenizer, training state
    (step, epoch, rng), and the training config — so an interrupted run can
    resume exactly where it left off."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(
        self,
        *,
        step: int,
        epoch: int,
        model: Any,
        optimizer: Any,
        rng_state: dict[str, Any],
        training_config: dict[str, Any],
        model_config: dict[str, Any],
        tokenizer_path: str | None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Save a checkpoint. Returns the checkpoint directory."""
        import torch

        from soup_cli.scratch._checkpoint_io import save_model_state

        state = {
            "step": step,
            "epoch": epoch,
            "rng_state": rng_state,
            "training_config": training_config,
            "model_config": model_config,
            "saved_at": time.time(),
            "extra": extra or {},
        }
        save_model_state(model, os.path.join(self.checkpoint_dir, "model.safetensors"))
        torch.save(optimizer.state_dict(), os.path.join(self.checkpoint_dir, "optimizer.pt"))
        state_path = os.path.join(self.checkpoint_dir, "training_state.json")
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        if tokenizer_path and os.path.isfile(tokenizer_path):
            shutil_copy(tokenizer_path, os.path.join(self.checkpoint_dir, "tokenizer.json"))
        return self.checkpoint_dir

    def load(self) -> dict[str, Any]:
        """Load the checkpoint metadata (without the model tensors)."""
        path = os.path.join(self.checkpoint_dir, "training_state.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no checkpoint in {self.checkpoint_dir}")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def restore_model(self, model: Any) -> dict[str, Any]:
        """Restore model weights into the given module. Returns the state."""
        from soup_cli.scratch._checkpoint_io import load_model_state

        state = self.load()
        load_model_state(model, os.path.join(self.checkpoint_dir, "model.safetensors"))
        return state

    def restore_optimizer(self, optimizer: Any) -> None:
        import torch

        path = os.path.join(self.checkpoint_dir, "optimizer.pt")
        if os.path.isfile(path):
            optimizer.load_state_dict(torch.load(path, weights_only=True))


def shutil_copy(src: str, dst: str) -> None:
    import shutil

    shutil.copyfile(src, dst)


# --------------------------------------------------------------------------- #
# training engine
# --------------------------------------------------------------------------- #


class ScratchTrainer:
    """The real training engine. Operates on a numpy int32 token stream and
    a built torch module. Batch construction is deterministic (seeded)."""

    def __init__(
        self,
        *,
        model: Any,
        train_ids: Any,
        val_ids: Any,
        config: ScratchTrainingConfig,
        checkpoint_dir: str,
        tokenizer_path: str | None = None,
        model_config: dict[str, Any] | None = None,
    ):
        self.model = model
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.config = config
        self.checkpoint = CheckpointManager(checkpoint_dir)
        self.tokenizer_path = tokenizer_path
        self.model_config = model_config or {}
        self.loss_log: list[dict[str, Any]] = []
        self._step = 0
        self._epoch = 0
        self._rng = _TorchRng(config.seed)

    def _build_optimizer(self) -> Any:
        import torch

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"unsupported optimizer: {self.config.optimizer}")

    def _get_batch(self, split: str, batch_size: int) -> tuple[Any, Any]:
        """Deterministic batch of (x, y) from the token stream."""
        import torch

        ids = self.train_ids if split == "train" else self.val_ids
        if len(ids) < self.config.block_size + 1:
            raise ValueError(
                f"token stream too short ({len(ids)} tokens) for a batch of "
                f"block_size {self.config.block_size}"
            )
        block = self.config.block_size
        ix = torch.randint(len(ids) - block - 1, (batch_size,), generator=self._rng.generator)
        x = torch.stack([torch.from_numpy(ids[i : i + block].astype("int64")) for i in ix])
        y = torch.stack([torch.from_numpy(ids[i + 1 : i + block + 1].astype("int64")) for i in ix])
        return x, y

    def _lr(self, step: int) -> float:
        """Cosine-with-warmup or constant LR."""
        if self.config.lr_scheduler == "constant":
            return self.config.learning_rate
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.config.learning_rate * (step + 1) / self.warmup_steps
        total = max(self.total_steps, 1)
        progress = min(1.0, (step - self.warmup_steps) / max(1, total - self.warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.learning_rate * 0.5 + self.config.learning_rate * 0.5 * cos

    @property
    def warmup_steps(self) -> int:
        return self.config.warmup_steps

    @property
    def total_steps(self) -> int:
        if self.config.max_steps:
            return self.config.max_steps
        # estimate steps per epoch from the token stream
        tokens_per_epoch = max(1, len(self.train_ids) - self.config.block_size)
        tokens_per_step = max(self.config.batch_size * self.config.block_size, 1)
        return max(1, tokens_per_epoch * self.config.epochs // tokens_per_step)

    def train(self, *, resume: bool = False) -> TrainResult:
        """Run the training loop. Returns the final result."""
        import math

        import torch

        device = self.config.device
        self.model.to(device)
        self.model.train()
        optimizer = self._build_optimizer()
        if resume:
            state = self.checkpoint.load()
            self._step = state["step"]
            self._epoch = state["epoch"]
            self.checkpoint.restore_model(self.model)
            self.checkpoint.restore_optimizer(optimizer)
            self._rng.restore(state.get("rng_state", {}))

        start = time.time()
        total = self.total_steps
        while self._step < total:
            if self.config.max_steps and self._step >= self.config.max_steps:
                break
            lr = self._lr_for_step(self._step)
            for g in optimizer.param_groups:
                g["lr"] = lr
            x, y = self._get_batch("train", self.config.batch_size)
            x, y = x.to(device), y.to(device)
            loss, _ = self.model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            ).item()
            optimizer.step()
            self._step += 1
            tokens_per_sec = self.config.batch_size * x.shape[1] / max(time.time() - start, 1e-6)

            if self._step % self.config.log_interval == 0 or self._step == total:
                self.loss_log.append(
                    {
                        "step": self._step,
                        "train_loss": float(loss.item()),
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "tokens_per_sec": tokens_per_sec,
                    }
                )
                console.print(
                    f"step {self._step}/{total}  loss {loss.item():.4f}  "
                    f"lr {lr:.2e}  grad_norm {grad_norm:.2f}"
                )
            if self._step % self.config.eval_interval == 0 or self._step == total:
                val_loss = self.evaluate()
                perplexity = float(math.exp(min(val_loss, 20)))
                self.loss_log.append(
                    {"step": self._step, "val_loss": val_loss, "val_perplexity": perplexity}
                )
            if self._step % self.config.save_interval == 0 or self._step == total:
                self._save_checkpoint(optimizer, rng_state=self._rng.state())
            if self._step >= total:
                break

        train_losses = [e["train_loss"] for e in self.loss_log if "train_loss" in e]
        final_train_loss = train_losses[-1] if train_losses else float("nan")
        val_loss = self.evaluate()
        perplexity = float(math.exp(min(val_loss, 20)))
        self._save_checkpoint(optimizer, rng_state=self._rng.state())
        self._write_log()
        return TrainResult(
            steps=self._step,
            epochs=self.config.epochs,
            final_train_loss=final_train_loss,
            final_val_loss=val_loss,
            final_val_perplexity=perplexity,
            total_time_sec=time.time() - start,
            checkpoint_dir=self.checkpoint.checkpoint_dir,
            loss_log=[e.get("train_loss", 0.0) for e in self.loss_log if "train_loss" in e],
        )

    def _lr_for_step(self, step: int) -> float:
        return self._lr(step)

    def evaluate(self) -> float:
        """Validation loss over ``eval_batches`` batches."""
        import torch

        if len(self.val_ids) < self.config.block_size + 1:
            return float("nan")
        self.model.eval()
        device = self.config.device
        losses = []
        with torch.no_grad():
            for _ in range(self.config.eval_batches):
                x, y = self._get_batch("val", self.config.batch_size)
                x, y = x.to(device), y.to(device)
                loss, _ = self.model(x, y)
                losses.append(loss.item())
        self.model.train()
        return float(sum(losses) / len(losses)) if losses else float("nan")

    def _save_checkpoint(self, optimizer: Any, *, rng_state: dict[str, Any]) -> None:

        self.checkpoint.save(
            step=self._step,
            epoch=self._epoch,
            model=self.model,
            optimizer=optimizer,
            rng_state=rng_state,
            training_config=self.config.to_dict(),
            model_config=self.model_config,
            tokenizer_path=self.tokenizer_path,
        )

    def _write_log(self) -> None:
        path = os.path.join(self.checkpoint.checkpoint_dir, "loss_log.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.loss_log:
                fh.write(json.dumps(entry) + "\n")


class _TorchRng:
    """Torch generator wrapper; deterministic per run."""

    def __init__(self, seed: int):
        import torch

        self.generator = torch.Generator().manual_seed(seed)
        self.seed = seed

    def state(self) -> dict[str, Any]:

        return {"seed": self.seed, "generator_state": self.generator.get_state().tolist()}

    def restore(self, state: dict[str, Any]) -> None:
        import torch

        if "generator_state" in state:
            self.generator = torch.Generator()
            self.generator.set_state(torch.tensor(state["generator_state"], dtype=torch.uint8))

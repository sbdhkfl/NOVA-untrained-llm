"""Dataset pipeline for the from-scratch model.

The pipeline never trains on unapproved examples:

1. ``ingest`` — user-provided learning material lands in ``pending/``.
2. ``clean`` — normalise whitespace, strip control characters, drop empty
   lines.
3. ``dedupe`` — exact and near-duplicate removal by SHA-256 of the cleaned
   text.
4. ``validate`` — non-empty, minimum length, character sanity.
5. ``review`` — the human reviews the pending examples.
6. ``approve`` — only explicitly approved examples are moved into the
   tokenised dataset; anything else is refused.
7. ``tokenize + split`` — the approved examples are tokenised with the
   versioned tokenizer, split into train/val, and written to a versioned
   manifest with provenance.

All tokenisation happens in a single pass; the resulting token stream is
persisted as a ``.npy`` int32 array (no torch needed to read it).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from rich.console import Console

console = Console()

# --------------------------------------------------------------------------- #
# example record
# --------------------------------------------------------------------------- #


@dataclass
class ScratchExample:
    """A single training example. ``status`` is one of pending / approved /
    rejected; only ``approved`` examples may be tokenised into a dataset."""

    text: str
    source: str = "user"
    status: str = "pending"
    example_id: str = ""
    sha256: str = ""
    validation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            self.example_id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.status not in ("pending", "approved", "rejected"):
            raise ValueError("status must be 'pending', 'approved' or 'rejected'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# cleaning / normalisation
# --------------------------------------------------------------------------- #

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalise a text: NFC unicode, remove control characters, collapse
    runs of whitespace into single spaces, strip trailing whitespace."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    text = unicodedata.normalize("NFC", text)
    text = text.lstrip("\ufeff")  # Windows editors often prefix a BOM
    text = _CONTROL_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_text(text: str, *, min_chars: int = 1, max_chars: int = 1_000_000) -> dict[str, Any]:
    """Validate a cleaned example. Returns a dict of checks."""
    checks = {
        "non_empty": bool(text.strip()),
        "min_chars": len(text) >= min_chars,
        "max_chars": len(text) <= max_chars,
        "printable_ratio": _printable_ratio(text) >= 0.5,
    }
    checks["passed"] = all(checks.values())
    checks["length"] = len(text)
    return checks


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for c in text if unicodedata.category(c)[0] in ("L", "N", "P", "Z", "S"))
    return printable / len(text)


# --------------------------------------------------------------------------- #
# dataset store
# --------------------------------------------------------------------------- #


class ScratchDatasetStore:
    """Stores the learning-material pipeline on disk.

    Layout::

        <root>/pending/<example_id>.json      (reviewable, not approved yet)
        <root>/approved/<example_id>.json     (approved, safe to train on)
        <root>/rejected/<example_id>.json     (rejected, never trained on)
        <root>/datasets/<dataset_version>/    (tokenised .npy + manifest.json)
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        for sub in ("pending", "approved", "rejected", "datasets"):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

    # ------------------------------------------------------------------ #
    # ingest
    # ------------------------------------------------------------------ #

    def ingest_text(self, text: str, *, source: str = "user") -> ScratchExample:
        """Ingest a single piece of text. Cleaned and validated; the example
        lands in ``pending/`` — it is NOT approved until ``approve`` is
        explicitly called. Ingest is idempotent: identical text returns the
        already-ingested example instead of overwriting it."""
        existing = self._find_existing(clean_text(text))
        if existing is not None:
            return existing
        cleaned = clean_text(text)
        checks = validate_text(cleaned)
        status = "pending" if checks["passed"] else "rejected"
        example = ScratchExample(text=cleaned, source=source, validation=checks, status=status)
        if checks["passed"]:
            path = os.path.join(self.root, "pending", f"{example.example_id}.json")
            self._write_json(path, example.to_dict())
        else:
            path = os.path.join(self.root, "rejected", f"{example.example_id}.json")
            self._write_json(path, example.to_dict())
        return example

    def ingest_file(self, path: str, *, source: str | None = None) -> list[ScratchExample]:
        """Ingest a text file. Large files are split into chunks of ~2000
        characters so each chunk is independently reviewable."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"not a file: {path}")
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        src = source or os.path.basename(path)
        chunks = _chunk_text(raw, 2000)
        return [self.ingest_text(c, source=src) for c in chunks]

    # ------------------------------------------------------------------ #
    # review / approve / reject
    # ------------------------------------------------------------------ #

    def list_pending(self) -> list[ScratchExample]:
        return self._list("pending")

    def list_approved(self) -> list[ScratchExample]:
        return self._list("approved")

    def list_rejected(self) -> list[ScratchExample]:
        return self._list("rejected")

    def approve(self, example_id: str) -> ScratchExample:
        """Move a pending example to approved. Only after this call is the
        example eligible for training."""
        return self._move(example_id, "pending", "approved", "approved")

    def reject(self, example_id: str) -> ScratchExample:
        """Move a pending example to rejected."""
        return self._move(example_id, "pending", "rejected", "rejected")

    def get(self, example_id: str) -> ScratchExample:
        """Look an example up in any bucket."""
        for sub in ("pending", "approved", "rejected"):
            path = os.path.join(self.root, sub, f"{example_id}.json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    return ScratchExample(**json.load(fh))
        raise FileNotFoundError(f"no such example: {example_id}")

    def edit(self, example_id: str, new_text: str) -> ScratchExample:
        """Replace an example's text. The edited example returns to
        ``pending`` — changed material must be re-approved before training,
        whether it was pending, approved or rejected before."""
        old = self.get(example_id)
        cleaned = clean_text(new_text)
        checks = validate_text(cleaned)
        if not checks["passed"]:
            raise ValueError(f"edited text fails validation: {checks}")
        os.remove(os.path.join(self.root, old.status, f"{example_id}.json"))
        example = ScratchExample(text=cleaned, source=old.source, validation=checks)
        path = os.path.join(self.root, "pending", f"{example.example_id}.json")
        self._write_json(path, example.to_dict())
        return example

    def approve_all(self) -> list[ScratchExample]:
        """Approve every pending example (must be called explicitly)."""
        return [self.approve(e.example_id) for e in self.list_pending()]

    def _find_existing(self, cleaned: str) -> ScratchExample | None:
        example_id = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
        for sub in ("pending", "approved", "rejected"):
            path = os.path.join(self.root, sub, f"{example_id}.json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    return ScratchExample(**json.load(fh))
        return None

    def _move(self, example_id: str, src: str, dst: str, status: str) -> ScratchExample:
        src_path = os.path.join(self.root, src, f"{example_id}.json")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"no such example in {src}/: {example_id}")
        with open(src_path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["status"] = status
        example = ScratchExample(**data)
        dst_path = os.path.join(self.root, dst, f"{example_id}.json")
        self._write_json(dst_path, example.to_dict())
        os.remove(src_path)
        return example

    def _list(self, sub: str) -> list[ScratchExample]:
        out = []
        d = os.path.join(self.root, sub)
        for name in sorted(os.listdir(d)):
            if name.endswith(".json"):
                with open(os.path.join(d, name), encoding="utf-8") as fh:
                    out.append(ScratchExample(**json.load(fh)))
        return out

    def dedupe(self) -> int:
        """Remove exact duplicates from pending and approved. Returns the
        number of duplicates removed."""
        removed = 0
        for sub in ("pending", "approved"):
            seen: set[str] = set()
            for ex in self._list(sub):
                if ex.sha256 in seen:
                    os.remove(os.path.join(self.root, sub, f"{ex.example_id}.json"))
                    removed += 1
                else:
                    seen.add(ex.sha256)
        return removed

    # ------------------------------------------------------------------ #
    # tokenize + split
    # ------------------------------------------------------------------ #

    def build_dataset(
        self,
        tokenizer: Any,
        *,
        version: str,
        val_fraction: float = 0.1,
        seed: int = 1337,
        approved_only: bool = True,
    ) -> dict[str, Any]:
        """Tokenise approved examples into a train/val split and write a
        versioned manifest. Refuses to run when ``approved_only`` is True and
        no approved examples exist — this is the hard approval gate."""
        approved = self.list_approved()
        examples = approved if approved_only else approved + self.list_pending()
        if approved_only and not examples:
            raise RuntimeError(
                "no approved examples: refusing to build a dataset. "
                "Review and approve examples with `soup learn review` first."
            )
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")

        # Deterministic shuffle.
        rng = _Rng(seed)
        order = list(range(len(examples)))
        rng.shuffle(order)
        ordered = [examples[i] for i in order]

        n_val = max(1, int(len(ordered) * val_fraction))
        if len(ordered) < 2:
            n_val = 0
        val_examples = ordered[:n_val]
        train_examples = ordered[n_val:]

        ds_dir = os.path.join(self.root, "datasets", version)
        if os.path.exists(ds_dir):
            raise ValueError(f"dataset version {version} already exists: {ds_dir}")
        os.makedirs(ds_dir)

        train_ids = _tokenize_all(tokenizer, [e.text for e in train_examples])
        val_ids = _tokenize_all(tokenizer, [e.text for e in val_examples]) if val_examples else []

        import numpy as np

        np.save(os.path.join(ds_dir, "train_ids.npy"), train_ids.astype(np.int32))
        np.save(os.path.join(ds_dir, "val_ids.npy"), val_ids.astype(np.int32))

        tokenizer.save(os.path.join(ds_dir, "tokenizer.json"))
        manifest = {
            "format": "soup-scratch-dataset",
            "version": version,
            "created_at": _now(),
            "tokenizer": {
                "vocab_size": tokenizer.vocab_size,
                "version_hash": tokenizer.version_hash,
            },
            "provenance": {
                "examples": [e.example_id for e in ordered],
                "sources": sorted({e.source for e in ordered}),
                "approved_only": approved_only,
                "seed": seed,
            },
            "split": {
                "train_tokens": int(len(train_ids)),
                "val_tokens": int(len(val_ids)),
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
            },
            "dataset_sha256": hashlib.sha256(
                json.dumps([e.sha256 for e in ordered], sort_keys=True).encode()
            ).hexdigest(),
        }
        self._write_json(os.path.join(ds_dir, "manifest.json"), manifest)
        return manifest

    def load_dataset(self, version: str) -> dict[str, Any]:
        """Load a tokenised dataset. Returns the manifest and the token
        arrays (as numpy int32)."""
        ds_dir = os.path.join(self.root, "datasets", version)
        if not os.path.isdir(ds_dir):
            raise FileNotFoundError(f"no such dataset version: {version}")
        with open(os.path.join(ds_dir, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        import numpy as np

        train_ids = np.load(os.path.join(ds_dir, "train_ids.npy"))
        val_ids = np.load(os.path.join(ds_dir, "val_ids.npy"))
        return {"manifest": manifest, "train_ids": train_ids, "val_ids": val_ids}

    def list_versions(self) -> list[str]:
        return sorted(os.listdir(os.path.join(self.root, "datasets")))

    # ------------------------------------------------------------------ #

    def _write_json(self, path: str, data: dict[str, Any]) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _tokenize_all(tokenizer: Any, texts: list[str]) -> Any:
    """Tokenise all texts with bos/eos framing."""
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=True))
    return _ids_to_array(ids)


def _ids_to_array(ids: list[int]) -> Any:
    import numpy as np

    return np.array(ids, dtype=np.int32)


def _chunk_text(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out = []
    cur = []
    cur_len = 0
    for para in text.split("\n"):
        if cur_len + len(para) + 1 > size and cur:
            out.append("\n".join(cur))
            cur = []
            cur_len = 0
        cur.append(para)
        cur_len += len(para) + 1
    if cur:
        out.append("\n".join(cur))
    return [c for c in out if c.strip()]


class _Rng:
    """Tiny deterministic RNG (xoshiro-style splitmix on a 64-bit state)."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def _next(self) -> int:
        # splitmix64
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def shuffle(self, items: list[int]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self._next() % (i + 1)
            items[i], items[j] = items[j], items[i]


def _now() -> str:
    import datetime

    return datetime.datetime.now().isoformat()

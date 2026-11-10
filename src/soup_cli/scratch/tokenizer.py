"""Byte-level BPE tokenizer for the from-scratch model.

Trained from the learning material itself, so the vocabulary reflects the
domain the model must learn. Requirements covered:

- consistent encode/decode (byte-level, so any text round-trips),
- special tokens (``<pad>``, ``<bos>``, ``<eos>``, ``<unk>``, user-defined),
- padding and sequence handling,
- reproducible tokenizer versions (versioned vocab + merges hash),
- vocabulary persistence (JSON),
- training/inference compatibility (same ``encode``/``decode`` on both sides).

The tokenizer is a pure-Python data structure; the training engine only
converts its output to tensors. No pretrained tokenizer is used.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Optional

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _byte_pair_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ScratchTokenizer:
    """Byte-level BPE tokenizer. Trained from the corpus by ``train()``.

    ``VOCAB`` is the byte-level base: 256 entries for the 256 bytes, then
    ``special tokens`` (``pad``/``bos``/``eos``/``unk`` and any extra ones),
    then ``merges``-derived tokens. All IDs are stable across save/load.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.special_tokens: list[str] = list(SPECIAL_TOKENS)
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {}
        self.inverse_vocab: dict[bytes, int] = {}
        # Special token ids are assigned deterministically: pad=256, bos=257,
        # eos=258, unk=259, then extras in order.
        self._special_ids: dict[str, int] = {}
        self._trained = False
        self._version = "1.0.0"
        self._version_hash: Optional[str] = None

    # ------------------------------------------------------------------ #
    # training / vocabulary construction
    # ------------------------------------------------------------------ #

    def train(self, texts: Iterable[str], *, progress: Any = None) -> None:
        """Train BPE merges from the corpus. Requires ``vocab_size`` to
        accommodate 256 byte tokens + 4 special tokens at minimum."""
        min_vocab = 256 + len(self.special_tokens)
        if self.vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size must be at least {min_vocab} (256 bytes + "
                f"{len(self.special_tokens)} special tokens)"
            )
        # Build the base vocabulary: 256 byte tokens, then special tokens.
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._special_ids = {}
        for i, tok in enumerate(self.special_tokens):
            self._special_ids[tok] = 256 + i
            self.vocab[256 + i] = tok.encode("utf-8")
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Concatenate the corpus at byte level.
        corpus: list[int] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"expected str, got {type(text).__name__}")
            corpus.extend(text.encode("utf-8"))
        if not corpus:
            raise ValueError("cannot train a tokenizer on an empty corpus")

        # Greedy BPE: most frequent pair becomes a new token each iteration.
        n_merges = self.vocab_size - min_vocab
        ids = list(corpus)
        for i in range(n_merges):
            stats = _byte_pair_stats(ids)
            if not stats:
                break
            best = max(stats, key=stats.get)
            if stats[best] < 2:
                break  # no pair worth merging
            new_id = min_vocab + i
            self.merges.append(best)
            ids = _merge(ids, best, new_id)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True
        self._version_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "merges": self.merges,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------ #
    # encoding / decoding
    # ------------------------------------------------------------------ #

    def _byte_to_id(self, b: int) -> int:
        """Each byte maps deterministically to its base id."""
        return b

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode a string into token ids. Special tokens are only added when
        ``add_special_tokens`` is True; unknown bytes are never emitted."""
        if not isinstance(text, str):
            raise TypeError(f"expected str, got {type(text).__name__}")
        if self._trained:
            ids = self._encode_bpe(text)
        else:
            ids = [self._byte_to_id(b) for b in text.encode("utf-8")]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def _encode_bpe(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        ids = [self._byte_to_id(b) for b in data]
        for pair in self.merges:
            ids = _merge(ids, pair, self._merge_id(pair))
        return ids

    def _merge_id(self, pair: tuple[int, int]) -> int:
        """The token id a merge produces (its position in the merge list)."""
        return 256 + len(self.special_tokens) + self.merges.index(pair)

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids back to a string. Special tokens are dropped
        unless ``skip_special_tokens`` is False. Byte-level BPE guarantees
        every text round-trips (lossless)."""
        out = []
        for i in ids:
            if i in self._special_ids.values():
                if not skip_special_tokens:
                    out.append(self.vocab[i])
                continue
            if i not in self.vocab:
                raise ValueError(f"unknown token id {i}")
            out.append(self.vocab[i])
        return b"".join(out).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # special tokens / padding
    # ------------------------------------------------------------------ #

    @property
    def pad_id(self) -> int:
        return self._special_ids["<pad>"]

    @property
    def bos_id(self) -> int:
        return self._special_ids["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._special_ids["<eos>"]

    @property
    def unk_id(self) -> int:
        return self._special_ids["<unk>"]

    def add_special_tokens(self, tokens: list[str]) -> None:
        """Register additional special tokens (e.g. user, assistant, system).
        Must be called before ``train()`` so the merge budget is reserved."""
        for tok in tokens:
            if tok not in self.special_tokens:
                self.special_tokens.append(tok)
        self._trained = False  # vocab must be retrained

    def pad(self, ids: list[int], max_len: int) -> list[int]:
        """Pad a sequence to ``max_len`` with the pad token id."""
        if len(ids) > max_len:
            return ids[:max_len]
        return ids + [self.pad_id] * (max_len - len(ids))

    def pad_batch(self, batch: list[list[int]], max_len: int | None = None) -> list[list[int]]:
        """Pad a whole batch to a common length."""
        target = max_len if max_len is not None else max(len(x) for x in batch)
        return [self.pad(x, target) for x in batch]

    # ------------------------------------------------------------------ #
    # persistence / versioning
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Persist the tokenizer to a versioned JSON file."""
        if not self._trained:
            raise RuntimeError("cannot save an untrained tokenizer")
        payload = {
            "format": "soup-scratch-bpe",
            "version": self._version,
            "version_hash": self._version_hash,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [list(p) for p in self.merges],
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
        }
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScratchTokenizer":
        """Load a persisted tokenizer. Raises if the file is not a soup-scratch
        BPE tokenizer (version mismatch)."""
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != "soup-scratch-bpe":
            raise ValueError(f"not a soup-scratch BPE tokenizer: {path}")
        tok = cls(vocab_size=payload["vocab_size"])
        tok.special_tokens = list(payload["special_tokens"])
        tok.merges = [tuple(p) for p in payload["merges"]]
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in payload["vocab"].items()}
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        tok._special_ids = {s: 256 + i for i, s in enumerate(tok.special_tokens)}
        tok._trained = True
        tok._version = payload["version"]
        tok._version_hash = payload["version_hash"]
        return tok

    @property
    def version(self) -> str:
        return self._version

    @property
    def version_hash(self) -> str:
        if self._version_hash is None:
            self._version_hash = self._compute_hash()
        return self._version_hash

    def token_stats(self) -> dict[str, Any]:
        """Statistics about the trained vocabulary."""
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": len(self.special_tokens),
            "merges": len(self.merges),
            "version_hash": self.version_hash,
        }

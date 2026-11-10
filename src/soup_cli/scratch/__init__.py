"""From-scratch training subsystem for SOUP.

A genuinely untrained language model that learns progressively from approved
training examples. This package provides:

- :mod:`soup_cli.scratch.model` — a small causal transformer built from scratch
  (token embedding, learned positional embeddings, causal multi-head attention,
  feed-forward layers, RMSNorm, LM output head). No pretrained weights.
- :mod:`soup_cli.scratch.tokenizer` — a byte-level BPE tokenizer trained from
  the learning material, with versioned vocabulary persistence.
- :mod:`soup_cli.scratch.dataset` — the dataset pipeline (ingest, clean,
  dedupe, validate, tokenize, split, versioned manifests with provenance).
- :mod:`soup_cli.scratch.engine` — the training engine (epochs, batches,
  optimizers, gradient clipping, validation, loss tracking, checkpoints,
  resume).
- :mod:`soup_cli.scratch.eval` — evaluation (loss, perplexity, reports).
- :mod:`soup_cli.scratch.remote` — remote training via Modal (never local).
- :mod:`soup_cli.scratch.learn` — the ``soup learn`` workflow orchestrator.
"""

from soup_cli.scratch.model import ScratchModelConfig, ScratchTransformer
from soup_cli.scratch.tokenizer import ScratchTokenizer

__all__ = ["ScratchModelConfig", "ScratchTransformer", "ScratchTokenizer"]

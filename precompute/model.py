"""MLX model wrapper: context -> next-token logits. Plumbing (not learning scope)."""
from __future__ import annotations

import mlx.core as mx
from mlx_lm import load

from typedefs import Logits, Context, Model, Tokenizer, Cache


def load_model(model_id: str) -> tuple[Model, Tokenizer]:
    """Load model + tokenizer via mlx-lm."""
    return load(model_id)


def encode(tokenizer: Tokenizer, text: str) -> Context:
    """Raw-text continuation: encode without a chat template."""
    return tokenizer.encode(text)


def next_token_logits(model: Model, token_ids: Context) -> Logits:
    """Full next-token logit vector for a context. Shape (vocab,)."""
    logits = model(mx.array(token_ids)[None])   # (1, seq, vocab)
    return logits[0, -1, :]


def next_token_logits_cached(model: Model, new_token_ids: Context, cache: Cache) -> Logits:
    """Forward only `new_token_ids` against an existing KV `cache` (extends it in place).
    Returns the last-position logits. The cache PLUMBING is here; the reuse STRATEGY
    (when to snapshot/copy it) is yours to implement in tree.py."""
    logits = model(mx.array(new_token_ids)[None], cache=cache)
    return logits[0, -1, :]

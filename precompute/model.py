"""MLX model wrapper: context -> next-token logits. Plumbing (not learning scope)."""
import mlx.core as mx
from mlx_lm import load


def load_model(model_id):
    """Load model + tokenizer via mlx-lm."""
    return load(model_id)


def encode(tokenizer, text):
    """Raw-text continuation: encode without a chat template."""
    return tokenizer.encode(text)


def next_token_logits(model, token_ids):
    """Full next-token logit vector for a context. Shape (vocab,)."""
    logits = model(mx.array(token_ids)[None])   # (1, seq, vocab)
    return logits[0, -1, :]


def next_token_logits_cached(model, new_token_ids, cache):
    """Forward only `new_token_ids` against an existing KV `cache` (extends it in place).
    Returns the last-position logits. The cache PLUMBING is here; the reuse STRATEGY
    (when to snapshot/copy it) is yours to implement in tree.py."""
    logits = model(mx.array(new_token_ids)[None], cache=cache)
    return logits[0, -1, :]

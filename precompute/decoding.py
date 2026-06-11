"""Decoding strategies + shared primitives (Decision 3).
Separate function per strategy; shared math in the helpers so it isn't duplicated.
Each operates on a single step's distribution."""
from __future__ import annotations

import mlx.core as mx

from typedefs import Logits, Probs, TokenId, TokenIds, Key


LOWER_BOUND = -1e9

# --- shared primitives ---
def softmax(logits: Logits) -> Probs:
    """Stable softmax -> probabilities (subtract max before exp)."""

    exp_sum = mx.sum(mx.exp(logits - logits[mx.argmax(logits)]))

    sftmx = mx.exp(logits - logits[mx.argmax(logits)])/exp_sum

    return sftmx

def apply_temperature(logits: Logits, t: float) -> Logits:
    """Return logits / t. t<1 sharpens toward greedy, t>1 flattens toward uniform."""
    return logits/t

# --- strategies ---
def greedy_select(logits: Logits) -> TokenId:
    """The argmax token id (deterministic). Equivalent to top_k_filter(., 1)."""
    return TokenId(mx.argmax(logits))

def top_k_filter(logits: Logits, k: int) -> TokenIds:
    """Indices of the top-k tokens — the candidate set top-k samples from (rigid; always k)."""
    return mx.argsort(-logits)[:k]


def top_p_filter(logits: Logits, p: float) -> TokenIds:
    """Smallest set of tokens whose cumulative prob >= p (the nucleus; includes the crossing token)."""

    probs_sorted = -1 * mx.sort(-softmax(logits))
    ind = mx.argsort(-logits)
    walking_sum = 0
    i = 0

    subset = []
    while walking_sum <= p and i < len(logits):
        subset.append(int(ind[i]))
        walking_sum += probs_sorted[i]
        i += 1

    return mx.array(subset)

def sample_from(logits: Logits, candidate_ids: TokenIds, rng: Key) -> TokenId:
    """Sample one id from candidate_ids, renormalized over just those (the dice step).
    Stochastic by design — that's what makes top-k/top-p *sampling* methods."""

    mask = mx.zeros(len(logits), dtype=mx.bool_)
    mask[candidate_ids] = True

    cnd_lgts = mx.where(mask, logits, LOWER_BOUND)

    return TokenId(mx.random.categorical(cnd_lgts, key=rng))

"""Decoding strategies + shared primitives (Decision 3).
Separate function per strategy; shared math in the helpers so it isn't duplicated.
Each operates on a single step's distribution."""
import mlx.core as mx


# --- shared primitives ---
def softmax(logits):
    """Stable softmax -> probabilities (subtract max before exp)."""
    # TODO(unnat): implement.
    raise NotImplementedError


def apply_temperature(logits, t):
    """Return logits / t. t<1 sharpens, t>1 flattens (claim #3). Consider t -> 0."""
    # TODO(unnat): implement — this is the whole of 'temperature' as a stage.
    raise NotImplementedError


# --- strategies ---
def greedy_select(logits):
    """The argmax token id (deterministic). Equivalent to top_k_filter(., 1)."""
    # TODO(unnat): implement.
    raise NotImplementedError


def top_k_filter(logits, k):
    """Indices of the top-k tokens — the candidate set top-k samples from (rigid; always k)."""
    # TODO(unnat): implement.
    raise NotImplementedError


def top_p_filter(logits, p):
    """Smallest set of tokens whose cumulative prob >= p (the nucleus; adaptive).
    Design qs: sort by prob desc, cumsum, cut at the crossing. Include the crossing token?
    (The nucleus paper includes it.)"""
    # TODO(unnat): implement nucleus selection.
    raise NotImplementedError


def sample_from(logits, candidate_ids, rng):
    """Sample one id from candidate_ids, renormalized over just those (the dice step).
    Stochastic by design — that's what makes top-k/top-p *sampling* methods."""
    # TODO(unnat): implement renormalize-over-candidates + sample.
    raise NotImplementedError

"""Decoding strategies + shared primitives (Decision 3).
Separate function per strategy; shared math in the helpers so it isn't duplicated.
Each operates on a single step's distribution."""
import mlx.core as mx


LOWER_BOUND = -1e-9

# --- shared primitives ---
def softmax(logits):
    """Stable softmax -> probabilities (subtract max before exp)."""

    exp_sum = mx.sum(mx.exp(logits - logits[mx.argmax(logits)]))

    sftmx = mx.exp(logits - logits[mx.argmax(logits)])/exp_sum

    return sftmx

def apply_temperature(logits, t):
    """Return logits / t. t<1 sharpens, t>1 flattens (claim #3). Consider t -> 0."""
    return logits/t

# --- strategies ---
def greedy_select(logits):
    """The argmax token id (deterministic). Equivalent to top_k_filter(., 1)."""
    return mx.argmax(logits)

def top_k_filter(logits, k):
    """Indices of the top-k tokens — the candidate set top-k samples from (rigid; always k)."""
    return mx.argsort(-logits)[:k]


def top_p_filter(logits, p):
    """Smallest set of tokens whose cumulative prob >= p (the nucleus; adaptive).
    Design qs: sort by prob desc, cumsum, cut at the crossing. Include the crossing token?
    (The nucleus paper includes it.)"""
    # TODO(unnat): implement nucleus selection.

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

def sample_from(logits, candidate_ids, rng):
    """Sample one id from candidate_ids, renormalized over just those (the dice step).
    Stochastic by design — that's what makes top-k/top-p *sampling* methods."""

    mask = mx.zeros(len(logits), dtype=mx.bool_)
    mask[candidate_ids] = True

    cnd_lgts = mx.where(mask, logits, LOWER_BOUND)

    return mx.random.categorical(cnd_lgts, key=rng)



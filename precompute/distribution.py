"""Per-node distribution representation (Decision 1 / ADR 0001).
Turns a full logit vector into the compact stored form: top-M (id, logit),
full-vocab logsumexp, tail residual mass, and entropy."""
import mlx.core as mx


def full_logsumexp(logits):
    """log(Σ exp(logits)) over the FULL vocab — the true softmax normalizer.
    Why: lets the head's probabilities be TRUE probabilities, not subset-renormalized.
    Watch numerical stability — subtract max(logits) before exp."""

    max_logits =  mx.max(logits) * mx.ones(len(logits))
    logits = mx.subtract(logits, max_logits)
    logit_sum = mx.sum(mx.exp(logits))

    normalized_sum = mx.log(logit_sum) 
    return mx.max(max_logits) + normalized_sum


def top_m(logits, m):
    """Top-m (token_id, logit) pairs, sorted descending by logit.
    Design q: argpartition vs argsort — does order within the top-m matter to you?
    Return: list[tuple[int, float]]."""
    sorted_logits_indices = mx.argsort(-logits)[:m]

    out = []
    for i in sorted_logits_indices:
        out.append((int(i), float(logits[i])))

    return out

def tail_mass(logits, kept_ids, lse):
    """Probability mass NOT covered by kept_ids: 1 - Σ prob(kept).
    prob_i = exp(logit_i - lse). Powers the 'other' bar + edge detection."""


    mask = mx.ones(len(logits), dtype=mx.bool_)
    mask[kept_ids] = False
    lse_arr = lse * mx.ones(len(logits))
    p = mx.exp(logits - lse_arr)
    p = mx.where(mask, p, 0.0)

    return mx.sum(p)



def entropy(logits, lse):
    """Shannon entropy of the FULL distribution, in BITS 
    H = -Σ p_i log2 p_i, with p_i = exp(logit_i - lse)."""
    # TODO(unnat): implement full-distribution entropy in bits.
    sum = 0
    lse_arr = lse * mx.ones(len(logits)) 
    p = mx.exp(logits - lse_arr)
    sum = mx.sum(p * mx.log2(p))
    return -1 * sum

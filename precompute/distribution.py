"""Per-node distribution representation (Decision 1 / ADR 0001).
Turns a full logit vector into the compact stored form: top-M (id, logit),
full-vocab logsumexp, tail residual mass, and entropy."""
import mlx.core as mx


def full_logsumexp(logits):
    """log(Σ exp(logits)) over the FULL vocab — the true softmax normalizer.
    Why: lets the head's probabilities be TRUE probabilities, not subset-renormalized.
    Watch numerical stability — subtract max(logits) before exp."""
    # TODO(unnat): implement numerically-stable logsumexp over all logits.
    raise NotImplementedError


def top_m(logits, m):
    """Top-m (token_id, logit) pairs, sorted descending by logit.
    Design q: argpartition vs argsort — does order within the top-m matter to you?
    Return: list[tuple[int, float]]."""
    # TODO(unnat): implement top-m selection.
    raise NotImplementedError


def tail_mass(logits, kept_ids, lse):
    """Probability mass NOT covered by kept_ids: 1 - Σ prob(kept).
    prob_i = exp(logit_i - lse). Powers the 'other' bar + edge detection."""
    # TODO(unnat): implement.
    raise NotImplementedError


def entropy(logits, lse):
    """Shannon entropy of the FULL distribution, in BITS (claim #4's number).
    H = -Σ p_i log2 p_i, with p_i = exp(logit_i - lse)."""
    # TODO(unnat): implement full-distribution entropy in bits.
    raise NotImplementedError

"""Tests for the per-node distribution representation.
One complete test shows the pattern; the rest are yours to implement (with the impl)."""
from distribution import full_logsumexp, top_m, tail_mass, entropy


def test_entropy_flat_exceeds_sharp(flat_logits, sharp_logits):   # COMPLETE — the pattern
    """A flat distribution must have higher entropy than a sharp one (claim #4)."""
    assert entropy(flat_logits, full_logsumexp(flat_logits)) > \
           entropy(sharp_logits, full_logsumexp(sharp_logits))


def test_top_m_returns_m_pairs_sorted_desc(sharp_logits):
    """top_m(logits, 3) returns 3 (id, logit) pairs, sorted by logit descending."""
    raise NotImplementedError


def test_tail_mass_plus_head_mass_is_one(sharp_logits):
    """tail_mass + sum(head probs over the stored top-M) approx 1.0."""
    raise NotImplementedError

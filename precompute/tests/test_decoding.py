"""Tests for decoding strategies + shared math.
One complete test shows the pattern; the rest are yours to implement (with the impl)."""
from decoding import softmax, apply_temperature, greedy_select, top_k_filter, top_p_filter


def test_softmax_sums_to_one_and_orders(sharp_logits):   # COMPLETE — the pattern
    p = softmax(sharp_logits)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert float(p[0]) > float(p[1])


def test_apply_temperature_flattens(flat_logits):
    """Higher temperature shrinks the gap between the top probabilities (claim #3)."""
    raise NotImplementedError


def test_top_k_keeps_exactly_k(sharp_logits):
    """top_k_filter(logits, k) returns exactly k indices — the k largest (claim #2: rigid)."""
    raise NotImplementedError


def test_top_p_wider_when_flat(flat_logits, sharp_logits):
    """At the same p, the flat distribution's nucleus has MORE tokens than the sharp one (claim #2: adaptive)."""
    raise NotImplementedError


def test_greedy_equals_top_k_one(sharp_logits):
    """greedy_select == the single index in top_k_filter(., 1) — they're the same thing."""
    raise NotImplementedError

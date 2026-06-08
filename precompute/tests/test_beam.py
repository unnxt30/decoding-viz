"""Tests for beam search. Stub — implement alongside beam.py.
A 2-layer hand-built fake `model` returning fixed logits is the cleanest way to test deterministically."""
from beam import beam_search


def test_beam_returns_width_hypotheses_ranked():
    """beam_search(..., width=4, ...) returns 4 hypotheses sorted by cumulative logprob desc.
    Each hypothesis: (token_ids: list[int], cum_logprob: float)."""
    raise NotImplementedError

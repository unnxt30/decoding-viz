"""Tests for beam search. Stub — implement alongside beam.py.
A 2-layer hand-built fake `model` returning fixed logits is the cleanest way to test deterministically."""
from beam import beam_search
import mlx.core as mx

class FakeModel:
    """Deterministic stand-in for an mlx-lm model.

      Contract (read model.next_token_logits): beam calls
          next_token_logits(model, ids) == model(mx.array(ids)[None])[0, -1, :]
      so __call__ receives x of shape (1, seq_len) and must return something
      indexable as [0, -1, :] giving a (vocab,) logit row → return (1, seq_len, vocab)."""

    def __init__(self, logits): 
        self.logits = mx.array(logits)

    def __call__(self, x):
        return mx.broadcast_to(self.logits, (1, x.shape[1], self.logits.shape[-1]))
        
        

def test_beam_returns_width_hypotheses_ranked():
    """beam_search(..., width=4, ...) returns 4 hypotheses sorted by cumulative logprob desc.
    Each hypothesis: (token_ids: list[int], cum_logprob: float)."""
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    model = FakeModel(logits)
    prompt_ids = [1, 2, 3]

    hypothesis = beam_search(model=model, prompt_ids=prompt_ids, horizon = 3, width=2)

    assert len(hypothesis) == 2

    assert hypothesis[0][0] == [7,7,7]
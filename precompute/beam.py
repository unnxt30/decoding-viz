"""Beam search (Decision 4): fixed width W, run to `horizon` tokens.
Keeps the W highest cumulative-log-prob SEQUENCES, not per-token picks."""
from __future__ import annotations

from typedefs import Model, Context, Hypothesis


def beam_search(model: Model, prompt_ids: Context, width: int, horizon: int) -> list[Hypothesis]:
    """Return the W final hypotheses: each (token_ids, cumulative_logprob).
    Algorithm (research 'beam search'):
      - maintain W live hypotheses (seq, score = Σ token logprobs)
      - each step: expand each hyp by its top candidates; score += logprob(next)
      - keep the global top-W across all expansions; stop at horizon (or EOS)
    Design qs: candidates-per-hyp before pruning (>= W)? length-normalize the score?
    how will these sequences map onto the precomputed node ids in export?"""
    # TODO(unnat): implement beam search to horizon.
    raise NotImplementedError

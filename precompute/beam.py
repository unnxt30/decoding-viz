"""Beam search (Decision 4): fixed width W, run to `horizon` tokens.
Keeps the W highest cumulative-log-prob SEQUENCES, not per-token picks."""
from __future__ import annotations

from typedefs import Model, Context, Hypothesis
from model import next_token_logits
from distribution import top_m, full_logsumexp
from typedefs import TokenId 
from decoding import softmax

hypothesis = []

def beam_search(model: Model, prompt_ids: Context, width: int, horizon: int) -> list[Hypothesis]:
    """Beam search to `horizon` tokens. Each step expands every live hypothesis by its top-`width`
    candidates, pools them all, and keeps the global top-`width` by cumulative log-prob.
    Returns the W final hypotheses, each (token_ids, cumulative_logprob)."""

    beam = [([], 0.0)]
    for _ in range(horizon):
        candidates = []

        for seq, score in beam:
            logits = next_token_logits(model, prompt_ids+seq)
            log_probs = logits - full_logsumexp(logits)


            for tok, logp in top_m(log_probs, width):
                candidates.append((seq + [tok], score + logp))


        beam = sorted(candidates, key=lambda x:x[1], reverse=True)[:width]
    
    return beam


# ---------------------------------------------------------------------------
# Learning note — how beam differs from what I tried first
#
# Wrong turn #1 — recursion per branch:
#     for tok in top_tokens: beam_search(prompt + [tok], ...)
#   Expands the top-W of EVERY node recursively -> width^horizon paths, never
#   pruned. That's a TREE (tree.py), not beam. Each branch lives in its own
#   call, so siblings are never in one place to be compared -> the prune is
#   structurally impossible.
#
# Wrong turn #2 — top-W openers, then greedy:
#   pick the top-W first tokens, then run GREEDY (top-1) from each to horizon.
#   = W independent greedy rollouts. Branches never compete after token 1, and
#   each node expands only its single best child (1 arrow, not W).
#
# What beam actually is (the d2l figure):
#   At EACH step, expand EVERY live hypothesis by its top-W candidates, pool
#   them ALL into one list, then keep the GLOBAL top-W by cumulative score
#   (the "circled" nodes). Both survivors can come from the same branch -> the
#   other line dies. That pool -> sort -> slice-to-width IS the prune, and it's
#   the exact thing both wrong turns lacked.
#
# Two enablers that make the prune work:
#   - Each hypothesis carries its OWN full token list (`seq`), so pooling across
#     branches is unambiguous — lineage is baked into the sequence, no links.
#   - Score is a sum of LOG-probabilities (logit - logsumexp) so it equals
#     log P(sequence); summing raw probs (or logits) gives the wrong ranking.
# ---------------------------------------------------------------------------
"""Shared type aliases for the precompute pipeline.

Intent aliases over mlx / builtin types. A type checker treats every `mx.array`
alias as identical, so these add no runtime enforcement — but they make each
function's CONTRACT explicit (a logit vector vs a probability vector vs a 0-d
scalar vs a token id), so the data flow reads at a glance and you stay honest.

(If you ever want the checker to actually enforce Logits-vs-Probs, switch these to
typing.NewType — but that forces wrapping every return value, which is a lot of
friction for a learning pass. Plain aliases give the intuition without the wrapping.)
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

# --- vocabulary-sized 1-D arrays, shape (vocab_size,) ---
Logits = mx.array      # raw, unbounded scores straight from the model
Probs = mx.array       # a probability distribution: non-negative, sums to 1
LogProbs = mx.array    # log-probabilities (logit - logsumexp)

# --- scalars: mlx reductions return 0-d arrays, not python floats ---
Scalar = mx.array      # a 0-d mlx array (an entropy value, a logsumexp, ...)

# --- token references ---
TokenId = int                  # one vocabulary id
TokenIds = mx.array            # a 1-D int array of ids (a candidate set, sorted indices)
Context = list[int]            # a sequence of token ids forming a model context

# --- opaque mlx / mlx-lm handles (named for readability; not structurally checked) ---
Model = Any                    # an mlx-lm model (nn.Module)
Tokenizer = Any                # an mlx-lm TokenizerWrapper
Cache = Any                    # an mlx-lm KV cache (list of per-layer caches)
Key = mx.array                 # an mlx PRNG key (mx.random.key(seed))

# --- beam search ---
Hypothesis = tuple[list[int], float]   # (token_ids_so_far, cumulative_logprob)

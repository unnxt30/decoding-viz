"""Automated prompt selection (Decision 7): score each candidate by fork-entropy and
greedy-loopiness, then rank and return the top `cfg.num_prompts`. Build-time curation only —
nothing downstream reads these scores.
"""
from __future__ import annotations

from statistics import mean, pstdev

from config import PrecomputeConfig
from typedefs import Model, Context, Tokenizer
from distribution import entropy, full_logsumexp
from mlx_lm.models.cache import make_prompt_cache
from model import next_token_logits_cached 
from decoding import greedy_select
from tree import extend_greedy


def fork_entropy_score(model: Model, prompt_ids: Context, lookahead: int) -> float:
    """Fork-demo score: the peak full-vocab entropy (bits) over the first `lookahead` greedy
    steps. Higher = a flatter step somewhere = strategies diverge more there. Build-time only."""

    ent = 0.0 
    cache = make_prompt_cache(model)
    seq = prompt_ids
    for i in range(lookahead):
        next_logits = next_token_logits_cached(model, seq, cache)
        seq = []
        tok = greedy_select(next_logits)
        ent = max(float(entropy(next_logits, full_logsumexp(next_logits))), ent)
        seq.append(tok)


    return float(ent)




def greedy_loopiness_score(greedy_tokens: list[int]) -> float:
    """Loopiness score for a greedy continuation: max(rep₂, rep₃) over the token ids, where
    repₙ = 1 − unique n-grams / total n-grams. Higher = more repetition. Pure function (no model)."""

    bi_gram, tri_gram = n_gram_count(greedy_tokens, 2), n_gram_count(greedy_tokens, 3)

    return float(max(bi_gram, tri_gram))


def n_gram_count(greedy_tokens: list[int], n: int=2) -> float:
    """repₙ over the token ids: 1 − unique n-grams / total n-grams (fraction of n-grams that recur)."""
    vocab = {}

    window_size = n  
    start = 0

    while start + window_size <= len(greedy_tokens):

        cand = tuple(greedy_tokens[start:start+window_size])
        vocab[cand]  = vocab.get(cand, 0) + 1
        start += 1
    

    total_n_grams = sum(vocab.values())
    unique = len(vocab)

    return 1 - unique/total_n_grams


def rank_prompts(model: Model, tokenizer: Tokenizer, candidates: list[str], cfg: PrecomputeConfig) -> list[str]:
    """Curate the corpus: score every candidate on fork-entropy + greedy-loopiness, z-score each
    signal across the set, combine (fork-weighted), then select `cfg.num_prompts` guaranteeing both
    a fork demo and a loop demo. Returns prompt strings (main.run re-encodes each when it builds)."""

    # ── stage 1 · GATHER (your loop) ──
    scored: list[tuple[str, float, float]] = []   # (candidate, fork_raw, loop_raw)

    for candidate in candidates:
        # tokenize the candidates first 
        candidates_tok = tokenizer.encode(candidate)

        next_toks = extend_greedy(model, candidates_tok, horizon=cfg.horizon)
        toks = [tok for tok, _ in next_toks]

        ent = fork_entropy_score(model, candidates_tok, lookahead=4)
        loopy_score = greedy_loopiness_score(toks)

        scored.append((candidate, ent, loopy_score))

    


    # ── stage 2 · NORMALIZE — z-score each signal ACROSS the candidate set ──
    # z-score (not min-max): we can't assume the score distribution up front, and centering on
    # the set's own mean/std keeps fork and loop comparable without a bounded-range assumption.
    # pstdev == 0 → the column carries no signal (all-equal, or a single candidate) → zeros,
    # not a divide-by-zero.
    if not scored:
        return []
    forks = [f for _, f, _ in scored]
    loops = [l for _, _, l in scored]

    def _z(xs: list[float]) -> list[float]:
        sd = pstdev(xs)
        if sd == 0:
            return [0.0] * len(xs)
        mu = mean(xs)
        return [(x - mu) / sd for x in xs]

    fz, lz = _z(forks), _z(loops)

    # ── stage 3 · COMBINE — fork weighted above loop (your call) ──
    W_FORK = 0.65   # forks are the rarer/harder demo to surface → weight them up; tweak freely
    combined = [W_FORK * f + (1 - W_FORK) * l for f, l in zip(fz, lz)]

    # ── stage 4 · SELECT for spread — guarantee BOTH a fork demo and a loop demo ──
    # Seed with the single strongest fork and strongest loop, then fill the rest by combined.
    # NOTE: argmax is invariant to monotonic rescaling, so the seeds are identical whether you
    # z-score, min-max, or use raw — normalization only moves the combined FILL ranking below.
    k = min(cfg.num_prompts, len(scored))
    forkiest = max(range(len(scored)), key=lambda i: fz[i])
    loopiest = max(range(len(scored)), key=lambda i: lz[i])

    chosen: list[int] = []
    for seed in (forkiest, loopiest):          # fork first → it survives when k == 1
        if seed not in chosen:
            chosen.append(seed)
    for idx in sorted(range(len(scored)), key=lambda i: combined[i], reverse=True):
        if len(chosen) >= k:
            break
        if idx not in chosen:
            chosen.append(idx)

    return [scored[i][0] for i in chosen[:k]]
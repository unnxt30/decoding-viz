"""Automated prompt selection (Decision 7): score candidates by how good a DEMO they make."""
from __future__ import annotations

from config import PrecomputeConfig
from typedefs import Model, Context


def fork_entropy_score(model: Model, prompt_ids: Context, lookahead: int) -> float:
    """How uncertain over the first `lookahead` steps? Higher = better fork demo.
    Design q: peak entropy across steps, or mean? (You want at least one genuinely flat step.)"""
    # TODO(unnat): implement.
    raise NotImplementedError


def greedy_loopiness_score(greedy_tokens: list[int]) -> float:
    """Does greedy repeat? Higher = better 'bland/loopy' demo (claim #1).
    Design q: repeated n-grams? distinct-token ratio?"""
    # TODO(unnat): implement a repetition metric.
    raise NotImplementedError


def rank_prompts(model: Model, candidates: list[str], cfg: PrecomputeConfig) -> list[str]:
    """Score every candidate, combine the two signals, return the top cfg.num_prompts.
    Design q: how to weight fork-entropy vs loopiness, and ensure a spread of both?"""
    # TODO(unnat): implement scoring + ranking + selection.
    raise NotImplementedError

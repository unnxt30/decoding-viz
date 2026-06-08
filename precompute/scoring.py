"""Automated prompt selection (Decision 7): score candidates by how good a DEMO they make."""


def fork_entropy_score(model, prompt_ids, lookahead):
    """How uncertain over the first `lookahead` steps? Higher = better fork demo.
    Design q: peak entropy across steps, or mean? (You want at least one genuinely flat step.)"""
    # TODO(unnat): implement.
    raise NotImplementedError


def greedy_loopiness_score(greedy_tokens):
    """Does greedy repeat? Higher = better 'bland/loopy' demo (claim #1).
    Design q: repeated n-grams? distinct-token ratio?"""
    # TODO(unnat): implement a repetition metric.
    raise NotImplementedError


def rank_prompts(model, candidates, cfg):
    """Score every candidate, combine the two signals, return the top cfg.num_prompts.
    Design q: how to weight fork-entropy vs loopiness, and ensure a spread of both?"""
    # TODO(unnat): implement scoring + ranking + selection.
    raise NotImplementedError

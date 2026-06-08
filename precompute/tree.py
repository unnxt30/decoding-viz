"""Build the bounded token-tree with KV-cache reuse (Decision 5) + the extended greedy path."""


def build_tree(model, prompt_ids, cfg):
    """Build the explorable tree: from the prompt, expand the top-N children to depth D,
    storing each node's distribution (distribution.py) at every node.

    KV-CACHE REUSE (your learning goal): descend depth-first; cache the prefix's KV so each
    child forwards only ONE new token (model.next_token_logits_cached). CRITICAL: siblings
    diverge, so you must SNAPSHOT/COPY the cache at each fork before descending a child —
    else sibling B inherits sibling A's extra token. Research: mlx_lm.models.cache
    (make_prompt_cache) and how to copy/trim cache state.

    Return: (nodes: dict[str, Node], root_id: str)."""
    # TODO(unnat): implement DFS tree build with cache snapshot at fork points.
    raise NotImplementedError


def extend_greedy(model, prompt_ids, horizon):
    """Pure greedy (argmax) out to `horizon` tokens — the bland/loopy evidence (claim #1).
    A single path: reuse one cache straight down (no snapshotting needed). Return list[(id, logprob)]."""
    # TODO(unnat): implement.
    raise NotImplementedError

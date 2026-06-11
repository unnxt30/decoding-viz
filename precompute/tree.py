"""Build the bounded token-tree with KV-cache reuse (Decision 5) + the extended greedy path."""
from __future__ import annotations

import schema
from config import PrecomputeConfig
from typedefs import Model, Context, TokenId, TokenIds, Logits, Scalar

from mlx_lm.models.cache import trim_prompt_cache, make_prompt_cache

from model import next_token_logits_cached 
from distribution import top_m, full_logsumexp, tail_mass, entropy 
from schema import Node 
from decoding import greedy_select

def build_tree(model: Model, prompt_ids: Context, cfg: PrecomputeConfig) -> tuple[dict[int, schema.Node], int]:
    """Build the explorable tree: from the prompt, expand the top-`branching_n` children to
    depth `cfg.tree_depth`, storing each node's distribution at every node.

    Depth-first with KV-cache reuse: each child forwards only one new token, and
    `trim_prompt_cache` rewinds the cache to the fork point after each child so siblings
    don't inherit each other's state. Returns (nodes: dict[int, Node], root_id: int)."""

    branch_factor = cfg.branching_n
    horizon = cfg.horizon
    m = cfg.top_m

    prompt_cache = make_prompt_cache(model)
    next_logits = next_token_logits_cached(model, prompt_ids, prompt_cache)

    nodes : dict[int, Node] = {}

    node_id = 0
    def expand(logits, depth, prior_token):
        nonlocal node_id
        top_m_logits = top_m(logits, m)
        lse = full_logsumexp(logits)
        kept_ids = [x for x, _ in top_m_logits] 
        tailmass = tail_mass(logits, TokenIds(kept_ids), lse)
        ent= entropy(logits, lse=lse)
        node_id += 1
        my_id = node_id
        node = Node(node_id=my_id, prior_token=prior_token, top_m_tokens=top_m_logits, lse=float(lse), tail_mass=float(tailmass), entropy=float(ent), edges = []) 

        nodes[node_id] = node

        if depth == cfg.tree_depth:
            return my_id
        else:
            fork = prompt_cache[0].offset
            top_n = top_m(logits, branch_factor)
            for i, _ in top_n:
                next_logits = next_token_logits_cached(model, [i], cache=prompt_cache)
                child_id = expand(next_logits, depth+1, i)
                node.edges.append(child_id) #type:ignore
                trim_prompt_cache(prompt_cache, prompt_cache[0].offset - fork)
            
            return my_id 

    root_id = expand(next_logits, depth=0, prior_token=prompt_ids[-1]) 
        
    return (nodes, root_id)





def extend_greedy(model: Model, prompt_ids: Context, horizon: int) -> list[tuple[TokenId, float]]:
    """Greedy (argmax) continuation out to `horizon` tokens, reusing one KV cache straight down.
    Returns [(token_id, logprob), ...]."""
    cache = make_prompt_cache(model)
    next_logits = next_token_logits_cached(model, prompt_ids, cache)

    ans = []
    for _ in range(horizon):
        greedy = greedy_select(next_logits)
        lse = full_logsumexp(next_logits)
        log_prob = next_logits - lse
        log_prob_tok = float(log_prob[greedy])
        next_logits = next_token_logits_cached(model, [greedy], cache)
        ans.append((greedy, log_prob_tok))

    return ans
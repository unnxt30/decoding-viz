"""Assemble precomputed structures into the JSON file + write it."""
from __future__ import annotations

import json
import dataclasses
from pathlib import Path

import schema
from config import PrecomputeConfig
from typedefs import Context, TokenId, Hypothesis, Tokenizer
from vocab import build_vocab_map


def assemble_prompt_file(prompt_text: str, prompt_ids: Context, tree: dict[int, schema.Node],
                         root_id: int, greedy: list[tuple[TokenId, float]],
                         beam: list[Hypothesis], tokenizer: Tokenizer,
                         cfg: PrecomputeConfig) -> schema.PromptFile:
    """Build the PromptFile (schema.py) from the precomputed pieces.
    Design: collect all referenced token ids -> build_vocab_map; assemble meta; pack nodes/paths."""
    # 1. Gather every token id the frontend might need a label for — from ALL sources
    #    (prompt, each node's prior_token + stored distribution, and both deterministic
    #    paths). Miss a source and that token renders as a bare id in the UI.
    referenced: set[TokenId] = set(prompt_ids)
    for node in tree.values():
        if node.prior_token is not None:
            referenced.add(node.prior_token)
        referenced.update(tid for tid, _ in (node.top_m_tokens or []))
    referenced.update(tid for tid, _ in greedy)
    for token_ids, _ in beam:                       # each Hypothesis is (token_ids, cum_logprob)
        referenced.update(token_ids)

    vocab = build_vocab_map(tokenizer, referenced)

    # 2. meta: the UI-relevant config — model badge + the slider caps the frontend must
    #    clamp to (ADR 0001). schema types meta as dict[str, str], so caps are stringified;
    #    loosen meta's value type if you'd rather keep them numeric.
    meta = {
        "model_id": cfg.model_id,
        "prompt_text": prompt_text,
        "top_p_max": str(cfg.top_p_max),
        "temp_max": str(cfg.temp_max),
        "branching_n": str(cfg.branching_n),
    }

    # 3. Pack the already-computed pieces into the data contract.
    return schema.PromptFile(
        root=root_id,
        nodes=tree,
        greedy_path=greedy,
        beam_path=beam,
        vocab=vocab,
        meta=meta,
    )


def _to_jsonable(o: object) -> dict:
    """json.dump fallback: recurse dataclass instances, else fall back to __dict__."""
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return o.__dict__


def write_prompt_file(prompt_file: schema.PromptFile, out_path: str | Path) -> None:
    """Serialize to JSON (plumbing)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(prompt_file, f, default=_to_jsonable)

"""Assemble precomputed structures into the JSON file + write it."""
from __future__ import annotations

import json
import dataclasses
from pathlib import Path

import schema
from config import PrecomputeConfig
from typedefs import Context, TokenId, Hypothesis, Tokenizer


def assemble_prompt_file(prompt_text: str, prompt_ids: Context, tree: dict[str, schema.Node],
                         root_id: str, greedy: list[tuple[TokenId, float]],
                         beam: list[Hypothesis], tokenizer: Tokenizer,
                         cfg: PrecomputeConfig) -> schema.PromptFile:
    """Build the PromptFile (schema.py) from the precomputed pieces.
    Design: collect all referenced token ids -> build_vocab_map; assemble meta; pack nodes/paths."""
    # TODO(unnat): implement assembly into the schema object.
    raise NotImplementedError


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

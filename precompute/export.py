"""Assemble precomputed structures into the JSON file + write it."""
import json
import dataclasses
from pathlib import Path


def assemble_prompt_file(prompt_text, prompt_ids, tree, root_id, greedy, beam, tokenizer, cfg):
    """Build the PromptFile (schema.py) from the precomputed pieces.
    Design: collect all referenced token ids -> build_vocab_map; assemble meta; pack nodes/paths."""
    # TODO(unnat): implement assembly into the schema object.
    raise NotImplementedError


def _to_jsonable(o):
    """json.dump fallback: recurse dataclass instances, else fall back to __dict__."""
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return o.__dict__


def write_prompt_file(prompt_file, out_path):
    """Serialize to JSON (plumbing)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(prompt_file, f, default=_to_jsonable)

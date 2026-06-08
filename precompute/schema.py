"""Data contract: the JSON shape the frontend consumes (spec §5/§6, Decision 6).
You DESIGN these fields. Read spec §6 for the agreed shape, then declare them.
Type aliases in typedefs.py (TokenId, etc.) are available if useful."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    """One tree node — the frontend draws bars + computes strategies from this.
    Design qs (spec §6 + ADR 0001):
      - the token that led here (how do you mark the root?)
      - the stored distribution (what element type — a small (id, logit) pair?)
      - the three honesty numbers (which? — ADR 0001)
      - how children are referenced (Decision 2 = by id)
      - is depth needed?
    Then pick types. Loose count: ~6-7 fields."""
    # TODO(unnat): declare Node fields.


@dataclass
class PromptFile:
    """The whole per-prompt JSON. Design qs: meta (which config values matter to the UI?),
    vocab map, the flat node map + root id, greedy_path, beam_paths. ~5-6 fields."""
    # TODO(unnat): declare PromptFile fields.

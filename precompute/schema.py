"""Data contract: the JSON shape the frontend consumes (spec §5/§6, Decision 6).
You DESIGN these fields. Read spec §6 for the agreed shape, then declare them.
Type aliases in typedefs.py (TokenId, etc.) are available if useful."""
from __future__ import annotations

from dataclasses import dataclass
from typedefs import TokenId, Hypothesis
from typing import Optional
@dataclass
class Node:
    """One tree node: the stored top-M distribution + honesty numbers (lse, tail_mass, entropy),
    the token that led here, and child node ids."""
    node_id: int
    prior_token: Optional[TokenId]
    top_m_tokens: Optional[list[tuple[TokenId, float]]]
    lse: Optional[float]
    tail_mass: Optional[float]
    entropy: Optional[float]
    edges: Optional[list[int]]

@dataclass
class PromptFile:
    """The whole per-prompt JSON contract the frontend consumes."""
    root: int
    nodes: dict[int, Node]
    greedy_path:list[tuple[TokenId, float]]
    beam_path:list[Hypothesis]
    vocab: dict[TokenId, str]
    meta: dict[str, str]

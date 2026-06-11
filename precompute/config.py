"""Precompute configuration — encodes the locked architecture decisions (spec §9)."""
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B-MLX-4bit"


@dataclass(frozen=True)
class PrecomputeConfig:
    model_id: str = MODEL_ID
    branching_n: int = 6        # top-N children expanded per node
    tree_depth: int = 4         # explorable tree depth D
    top_m: int = 100            # stored distribution size per node (Decision 1)
    beam_width: int = 4         # Decision 4
    horizon: int = 12           # greedy/beam deterministic path length
    top_p_max: float = 0.95     # slider cap (prevent — Decision 1)
    temp_max: float = 1.5       # slider cap
    num_prompts: int = 6        # selected by automated search (Decision 7)
    out_dir: Path = field(default_factory=lambda: Path("../site/data"))

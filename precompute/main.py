"""CLI: automated search -> build trees -> export JSON for the top prompts."""
from __future__ import annotations

import config
import model as model_mod
import scoring
import tree
import beam
import export
from config import PrecomputeConfig


def run(cfg: PrecomputeConfig) -> None:
    """Full precompute (spec §11):
      1. load model
      2. rank_prompts over data/candidate_prompts.json -> top cfg.num_prompts
      3. each: build_tree + extend_greedy + beam_search -> assemble -> write JSON to cfg.out_dir"""
    # TODO(unnat): implement the orchestration.
    raise NotImplementedError


if __name__ == "__main__":
    run(config.PrecomputeConfig())

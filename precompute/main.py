"""CLI: automated search -> build trees -> export JSON for the top prompts."""
from __future__ import annotations


from mlx_lm import load
import config
import model as model_mod 
from scoring import rank_prompts
from tree import build_tree, extend_greedy
from beam import beam_search
from export import assemble_prompt_file, write_prompt_file
from config import PrecomputeConfig, MODEL_ID


def run(cfg: PrecomputeConfig) -> None:
    """Full precompute (spec §11):
      1. load model
      2. rank_prompts over data/candidate_prompts.json -> top cfg.num_prompts
      3. each: build_tree + extend_greedy + beam_search -> assemble -> write JSON to cfg.out_dir"""
    
    model, tokenizer = load(MODEL_ID) #pyright: ignore
    import json
    candidates = []
    with open("data/candidate_prompts.json", "r") as f:
        candidates = json.load(f)["prompts"]

    prompts = rank_prompts(model, tokenizer, candidates, cfg)

    for i,prmpt in enumerate(prompts):
        toks = model_mod.encode(tokenizer, prmpt) #type: ignore
        tree,root = build_tree(model, toks, cfg)
        greedy_pth = extend_greedy(model, toks, cfg.horizon)
        beam_path = beam_search(model, toks, cfg.beam_width, cfg.horizon)
        pf = assemble_prompt_file(prmpt, toks, tree, root, greedy_pth, beam_path, tokenizer, cfg)
        write_prompt_file(pf, out_path=f"{cfg.out_dir}/prompt{i}.json")

if __name__ == "__main__":
    run(config.PrecomputeConfig())

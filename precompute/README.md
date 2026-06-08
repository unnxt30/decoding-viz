# precompute

MLX pipeline that runs `Qwen3-4B-MLX-4bit` once over curated prompts and emits the per-prompt
JSON the static site consumes. **Learning mode:** the decoding logic here is yours to implement —
files are scaffolded with `# TODO(unnat)` stubs and `raise NotImplementedError`. See the plan
(`docs/superpowers/plans/2026-06-08-decoding-visualizer-plan.md`) and the locked decisions
(spec §9 + `docs/superpowers/adr/0001`).

## Setup
```bash
cd precompute
uv sync                 # installs mlx, mlx-lm, numpy, pytest
```

## Workflow (implement in this order — see the plan)
1. `distribution.py` — per-node representation (Decision 1)
2. `decoding.py` — strategies + shared math (Decision 3)
3. `beam.py` — beam search (Decision 4)
4. `tree.py` — KV-cache tree build (Decision 5)
5. `scoring.py` — automated prompt selection (Decision 7)
6. `vocab.py` / `schema.py` / `export.py` — output (Decision 6)
7. `main.py` — orchestrate end-to-end

## Run
```bash
uv run pytest                 # unit tests (red until you implement)
uv run python main.py         # full precompute -> ../site/data/*.json
```

`model.py` (MLX forward-pass → logits) is written for you — it's plumbing, not the learning surface.

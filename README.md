# decoding-visualizer

An interactive visualizer showing a real model's next-token distribution and how greedy / beam /
top-k / top-p / temperature actually pick tokens — built as evidence for a blog post on decoding
strategies.

- **`precompute/`** — MLX pipeline (`Qwen3-4B-MLX-4bit`) that runs once and emits per-prompt JSON.
  Built in **learning mode**: the decoding logic is scaffolded with `# TODO(unnat)` stubs for you
  to implement. See `precompute/README.md`.
- **`site/`** — static, client-side-interactive frontend that loads the JSON. Built by the agent.
- **`deploy.sh`** — `vercel --prod` from `site/` → a shareable `*.vercel.app` URL for the blog.

## Docs
- Spec: `docs/superpowers/specs/2026-06-08-decoding-visualizer-design.md`
- Decisions: `docs/superpowers/adr/0001-per-node-distribution-storage.md` (+ spec §9)
- Plan: `docs/superpowers/plans/2026-06-08-decoding-visualizer-plan.md`

## Quick start
```bash
cd precompute && uv sync          # set up the precompute env
uv run pytest                     # decoding tests — red until you implement
# ... implement the TODO(unnat) bodies, then:
uv run python main.py             # -> ../site/data/*.json
cd .. && ./deploy.sh              # -> live URL
```

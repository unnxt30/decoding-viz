// Data layer — fetch the manifest + a prompt's JSON, expose lookups. Pure plumbing (no decoding math).
// Matches the precompute schema (precompute/schema.py): PromptFile { root, nodes, greedy_path,
// beam_path, vocab, meta }, Node { node_id, prior_token, top_m_tokens:[[id,logit]], lse, tail_mass,
// entropy, edges:[childId] }. JSON object keys are strings, so node/vocab ids are looked up via String().

/** The curated-prompt manifest: [{ file, text }, ...] (built from the 6 prompt files). */
export async function loadManifest() {
  const res = await fetch("data/index.json");
  if (!res.ok) throw new Error(`manifest load failed: ${res.status}`);
  return (await res.json()).prompts;
}

/** Load one per-prompt file and wrap it for the UI. */
export async function loadPromptFile(file) {
  const res = await fetch(`data/${file}`);
  if (!res.ok) throw new Error(`failed to load data/${file}: ${res.status}`);
  return new PromptData(await res.json());
}

/** Thin, typed view over the raw JSON: node lookup by id, display strings, deterministic paths. */
export class PromptData {
  constructor(file) {
    this.meta = file.meta;                 // { model_id, prompt_text, top_p_max, temp_max, branching_n }
    this.vocab = file.vocab;               // { "id": "display" }  (only referenced ids)
    this.root = file.root;                 // int
    this.nodes = file.nodes;               // { "id": Node }
    this.greedyPath = file.greedy_path;    // [[id, logprob], ...]   length = horizon
    this.beamPath = file.beam_path;        // [[[id, ...], cumLogprob], ...]   W hypotheses
  }

  node(id) { return this.nodes[String(id)]; }
  display(id) { return this.vocab[String(id)] ?? `⟨${id}⟩`; }
  /** child node ids of a node (the expanded branches). */
  edges(id) { return this.node(id)?.edges ?? []; }
  /** decode a list of token ids to a single display string. */
  text(ids) { return ids.map((i) => this.display(i)).join(""); }

  get tempMax() { return parseFloat(this.meta.temp_max); }
  get topPMax() { return parseFloat(this.meta.top_p_max); }
  /** k is capped at the stored distribution size (ADR 0001: never ask for more than top-M). */
  get topMSize() { return this.node(this.root)?.top_m_tokens?.length ?? 0; }
  get modelBadge() { return this.meta.model_id; }
}

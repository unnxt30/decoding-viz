// Data layer — fetch a prompt's JSON and expose lookups. Pure plumbing (no decoding math).

/** Load a per-prompt file from /data/<name>.json and index it for the UI. */
export async function loadPromptFile(name) {
  const res = await fetch(`data/${name}.json`);
  if (!res.ok) throw new Error(`failed to load data/${name}.json: ${res.status}`);
  const file = await res.json();
  return new PromptData(file);
}

/** Thin wrapper over the raw JSON: node lookup by id, display strings, paths. */
export class PromptData {
  constructor(file) {
    this.meta = file.meta;
    this.vocab = file.vocab;            // { id: displayString }
    this.root = file.root;
    this.nodes = file.nodes;            // { id: node }
    this.greedyPath = file.greedy_path; // [id, ...]
    this.beamPaths = file.beam_paths;   // [{ node_ids, cum_logprob }]
  }
  node(id) { return this.nodes[id]; }
  display(tokenId) { return this.vocab[String(tokenId)] ?? `⟨${tokenId}⟩`; }
  children(id) { return this.node(id)?.children ?? {}; }
}

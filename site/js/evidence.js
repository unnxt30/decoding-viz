// Evidence mode (bottom): the same prompt run through every strategy, side by side — the figure
// the blog screenshots. Deterministic rows (greedy, beam) come straight from the precomputed paths.
// Sampling rows roll live in the browser, walking DOWN the tree (so they stay inside precomputed
// bounds → depth-limited), re-rollable. Exports sampleStep() so the 🎲 dice in main.js reuses it.

import { nodeDistribution, topKSet, topPSet } from "./strategies.js";

const escHtml = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const escWs = (s) => escHtml(s).replace(/␣/g, '<span class="ws">␣</span>').replace(/↵/g, '<span class="ws">↵</span>');

function rep2(ids) {
  if (ids.length < 2) return 0;
  const bg = [];
  for (let i = 0; i < ids.length - 1; i++) bg.push(ids[i] + "," + ids[i + 1]);
  return 1 - new Set(bg).size / bg.length;
}

/** Branched children of a node as {childId, id(token), p}, prob from the node's dist at temp. */
function childPool(data, nodeId, temp) {
  const node = data.node(nodeId);
  if (!node) return [];
  const { dist } = nodeDistribution(node, temp);
  const pmap = new Map(dist.map((d) => [d.id, d.p]));
  return data.edges(nodeId).map((cid) => {
    const c = data.node(cid);
    return { childId: cid, id: c.prior_token, p: pmap.get(c.prior_token) ?? 0 };
  });
}

function sampleWeighted(items) {
  const z = items.reduce((a, b) => a + b.p, 0) || 1;
  let r = Math.random() * z;
  for (const it of items) { r -= it.p; if (r <= 0) return it; }
  return items[items.length - 1];
}

/** One in-tree sampled step from `nodeId` under params. Returns {childId, id} or null (leaf). */
export function sampleStep(data, nodeId, { strategy, temp, k, p }) {
  let pool = childPool(data, nodeId, temp);
  if (!pool.length) return null;
  if (strategy === "topk" || strategy === "topp") {
    const { dist } = nodeDistribution(data.node(nodeId), temp);
    const set = strategy === "topk" ? topKSet(dist, k) : topPSet(dist, p);
    const filtered = pool.filter((c) => set.has(c.id));
    if (filtered.length) pool = filtered; // else fall back to all branched kids
  }
  return sampleWeighted(pool);
}

/** Walk down the tree sampling each step until a leaf (depth-bounded). Returns token ids. */
function walk(data, params) {
  let id = data.root;
  const ids = [];
  for (let s = 0; s < 8; s++) {
    const step = sampleStep(data, id, params);
    if (!step) break;
    ids.push(step.id);
    id = step.childId;
  }
  return ids;
}

function row(name, tag, dotVar, seed, gen, loopy) {
  return `<div class="ev-row">
    <div class="ev-strat">
      <span class="nm"><span class="dot" style="background:var(${dotVar})"></span>${name}</span>
      <span class="tag">${tag}</span>
    </div>
    <div class="ev-cont ${loopy ? "loopy" : ""}"><span class="seed">${escWs(seed)}</span><span class="gen">${escWs(gen)}</span></div>
  </div>`;
}

function sampledRow(data, name, tag, params) {
  const ids = walk(data, params);
  return row(name, tag, "--green", data.meta.prompt_text, data.text(ids), false);
}

export function renderEvidence(container, data) {
  const seed = data.meta.prompt_text;
  const greedyIds = data.greedyPath.map(([id]) => id);
  const beamBest = data.beamPath?.[0]?.[0] ?? [];

  const rows = [
    row("Greedy", "argmax · horizon 12", "--cyan", seed, data.text(greedyIds), rep2(greedyIds) >= 0.4),
    row("Beam (W=4)", "best of 4 · horizon 12", "--cyan", seed, data.text(beamBest), rep2(beamBest) >= 0.4),
    sampledRow(data, "Top-k", "k=10 · sampled, in-tree", { strategy: "topk", temp: 1, k: 10, p: 0.9 }),
    sampledRow(data, "Top-p", "p=0.90 · sampled, in-tree", { strategy: "topp", temp: 1, k: 10, p: 0.9 }),
    sampledRow(data, "High temp", `T=${data.tempMax} · sampled, in-tree`, { strategy: "greedy", temp: data.tempMax, k: 10, p: 0.9 }),
  ];

  container.innerHTML =
    rows.join("") +
    `<div style="text-align:right;margin-top:12px"><button class="dice" id="ev-reroll" style="width:auto;padding:7px 16px">↻ reroll samples</button></div>`;
  const btn = container.querySelector("#ev-reroll");
  if (btn) btn.addEventListener("click", () => renderEvidence(container, data));
}

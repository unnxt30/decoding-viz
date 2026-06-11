// Token-tree SVG render (Explore mode). The full tree is 1,555 nodes, so we don't draw it all —
// we render the current PATH as a spine and fan out each node's children, drilling down on click.
// current path highlighted, fork nodes (high entropy) ringed amber, node size ∝ edge probability.

import { nodeDistribution } from "./strategies.js";

const W = 760, H = 440, PADX = 64, PADY = 26;
const FORK_BITS = 3.0; // entropy at/above this rings the node as a fork

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const clip = (s, n = 9) => (s.length > n ? s.slice(0, n) + "…" : s);

/**
 * @param svg       the <svg> element
 * @param data      PromptData
 * @param path      [rootId, ...selectedAncestors, selectedId] — the spine to render
 * @param onSelect  (nodeId, columnIndex) => void  (column the node was clicked in)
 */
export function renderTree(svg, data, path, onSelect) {
  const selected = path[path.length - 1];

  // Build columns: col 0 = [root]; col c = children (edges) of path[c-1].
  const cols = [[data.root]];
  for (let c = 1; c <= path.length; c++) cols[c] = data.edges(path[c - 1]).slice();
  while (cols.length > 1 && cols[cols.length - 1].length === 0) cols.pop(); // drop empty trailing col

  const numCols = cols.length;
  const colX = (c) => (numCols === 1 ? W / 2 : PADX + (c * (W - 2 * PADX)) / (numCols - 1));
  const midY = H / 2;

  // Lay out node positions; parents are always in an earlier column so their pos exists first.
  const pos = {};
  const meta = {}; // id -> {p, entropy, prior}
  cols.forEach((ids, c) => {
    const n = ids.length;
    const gap = n > 1 ? Math.min(62, (H - 2 * PADY) / (n - 1)) : 0;
    const parentId = c === 0 ? null : path[c - 1];
    const pdist = parentId != null ? nodeDistribution(data.node(parentId)).dist : null;
    const pmap = pdist ? new Map(pdist.map((d) => [d.id, d.p])) : null;
    ids.forEach((id, i) => {
      const node = data.node(id);
      const prior = node?.prior_token;
      const p = c === 0 ? 1 : (pmap?.get(prior) ?? 0);
      const r = c === 0 ? 17 : 7 + 13 * Math.sqrt(Math.max(0, p));
      const y = midY + (i - (n - 1) / 2) * gap;
      pos[id] = { x: colX(c), y, r, c };
      meta[id] = { p, entropy: node?.entropy ?? 0, prior };
    });
  });

  const pathSet = new Set(path);
  let edges = "", nodes = "";

  // edges: parent (path[c-1]) -> each child in column c
  for (let c = 1; c < numCols; c++) {
    const parentId = path[c - 1];
    const pp = pos[parentId];
    if (!pp) continue;
    for (const id of cols[c]) {
      const cp = pos[id];
      const on = path[c] === id ? "on" : "";
      const mx = (pp.x + cp.x) / 2;
      edges += `<path class="tree-edge ${on}" d="M${pp.x + pp.r},${pp.y} C${mx},${pp.y} ${mx},${cp.y} ${cp.x - cp.r},${cp.y}"/>`;
    }
  }

  // nodes
  cols.forEach((ids, c) => {
    for (const id of ids) {
      const p = pos[id], m = meta[id];
      const isSel = id === selected;
      const onPath = pathSet.has(id);
      const fork = m.entropy >= FORK_BITS;
      const cls = ["tree-node"];
      if (isSel) cls.push("sel"); else if (onPath) cls.push("on");
      if (fork) cls.push("fork", "ring");
      const label = c === 0 ? "⟨prompt⟩" : clip(data.display(m.prior));
      const anchor = c === numCols - 1 ? "end" : "start";
      const tx = anchor === "end" ? p.x - p.r - 6 : p.x + p.r + 6;
      nodes += `<g class="${cls.join(" ")}" data-id="${id}" data-col="${c}">
        <circle cx="${p.x}" cy="${p.y}" r="${p.r.toFixed(1)}"></circle>
        <text x="${tx}" y="${(p.y + 4).toFixed(1)}" text-anchor="${anchor}">${esc(label)}</text>
        ${fork ? `<text class="tree-forkbadge" x="${p.x}" y="${(p.y - p.r - 4).toFixed(1)}" text-anchor="middle">⚡${m.entropy.toFixed(1)}</text>` : ""}
      </g>`;
    }
  });

  svg.innerHTML = `${edges}${nodes}
    <text class="tree-legend" x="${PADX}" y="${H - 6}">depth → · node size ∝ P(token) · ⚡ = entropy (bits)</text>`;

  svg.querySelectorAll(".tree-node").forEach((g) =>
    g.addEventListener("click", () => onSelect(g.dataset.id, +g.dataset.col))
  );
}

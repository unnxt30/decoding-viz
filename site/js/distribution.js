// Distribution panel (right top). Draws the selected node's stored top-M as sorted bars and
// applies the live strategy: top-p nucleus / top-k set highlighted green, cut tokens greyed,
// greedy pick marked, plus entropy / kept-count / "other" mass readout. Re-renders on every
// slider change — all math is client-side via strategies.js.

import { nodeDistribution, candidateSet, greedyId } from "./strategies.js";

const TOP_SHOWN = 14;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function renderDistribution(barsEl, metaEl, data, nodeId, params) {
  const node = data.node(nodeId);
  if (!node) { barsEl.innerHTML = `<div class="hint">no distribution for this node</div>`; metaEl.innerHTML = ""; return; }

  const { dist, tail } = nodeDistribution(node, params.temp);
  const gid = greedyId(dist);
  const { candidates, kept } = candidateSet(dist, params.strategy, params);

  const shown = dist.slice(0, TOP_SHOWN);
  const maxP = shown[0]?.p || 1;

  let rows = "";
  shown.forEach((d, i) => {
    const inSet = candidates ? candidates.has(d.id) : true;
    const fillCls = candidates ? (inSet ? "keep" : "cut") : "plain";
    const rowCls = (candidates && !inSet ? "cut " : "") + (d.id === gid ? "greedy " : "");
    const w = Math.max(2, (d.p / maxP) * 100);
    const tok = esc(data.display(d.id));
    rows += `<div class="bar-row ${rowCls}">
      <div class="bar-tok" title="${tok}">${tok}</div>
      <div class="bar-track"><div class="bar-fill ${fillCls}" style="width:${w.toFixed(1)}%"></div></div>
      <div class="bar-p">${(d.p * 100).toFixed(d.p < 0.001 ? 3 : 1)}%</div>
    </div>`;

    // one cutoff divider, right after the last kept token (top-k boundary / nucleus edge)
    if (candidates && i + 1 === kept && kept < shown.length) {
      const lbl = params.strategy === "topk" ? `k = ${kept}` : `nucleus · ${kept} tokens ≥ ${(params.p).toFixed(2)}`;
      rows += `<div class="cutdivider">— ${lbl} —</div>`;
    }
  });

  // residual "other" mass bar
  const otherW = Math.max(1.5, (tail / maxP) * 100);
  const other = `<div class="dist-other">
    <div class="bar-tok" style="color:var(--ink-faint)">other</div>
    <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, otherW).toFixed(1)}%"></div></div>
    <div class="bar-p">${(tail * 100).toFixed(2)}%</div>
  </div>`;

  barsEl.innerHTML = rows + other;

  metaEl.innerHTML = `
    <div class="stat ent"><div class="k">entropy</div><div class="v">${(node.entropy ?? 0).toFixed(2)}<span class="hint"> bits</span></div></div>
    <div class="stat kept"><div class="k">${params.strategy === "topp" ? "nucleus" : params.strategy === "topk" ? "kept (k)" : "candidates"}</div><div class="v">${candidates ? kept : "all"}</div></div>
    <div class="stat other"><div class="k">other mass</div><div class="v">${(tail * 100).toFixed(1)}<span class="hint">%</span></div></div>`;
}

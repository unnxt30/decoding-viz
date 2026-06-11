// App wiring: manifest → prompt picker → tree / distribution / controls, with drill-down
// navigation, the 🎲 dice (sample a step + descend), and the Explore/Evidence mode toggle.
// All decoding math is client-side (strategies.js mirrors the Python core). No backend at runtime.

import { loadManifest, loadPromptFile } from "./data.js";
import { renderTree } from "./tree.js";
import { renderDistribution } from "./distribution.js";
import { renderControls } from "./controls.js";
import { renderEvidence, sampleStep } from "./evidence.js";

const $ = (id) => document.getElementById(id);
const escHtml = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const escAttr = (s) => escHtml(s).replace(/"/g, "&quot;");

let data = null;
let path = [];                                  // node ids root → selected (all numeric)
let mode = "explore";
const params = { strategy: "greedy", temp: 1.0, k: 10, p: 0.9 };
const selId = () => path[path.length - 1];

async function init() {
  const manifest = await loadManifest();
  const picker = $("prompt-picker");
  picker.innerHTML = manifest.map((m) => `<option value="${m.file}">${escAttr(m.text)}</option>`).join("");
  picker.addEventListener("change", () => loadPrompt(picker.value));
  setupModes();
  $("dice").addEventListener("click", rollDice);
  await loadPrompt(manifest[0].file);
}

async function loadPrompt(file) {
  data = await loadPromptFile(file);
  $("model-badge").textContent = data.modelBadge;
  params.k = Math.min(params.k, data.topMSize || params.k);
  path = [data.root];
  $("dice-out").textContent = "";
  renderAll();
}

function renderAll() {
  renderTree($("tree-svg"), data, path, onTreeSelect);
  renderControls($("strategy-tabs"), $("sliders"), data, params, onControlsChange);
  renderDist();
  if (mode === "evidence") renderEvidence($("evidence-rows"), data);
}

function renderDist() {
  const id = selId();
  const node = data.node(id);
  $("dist-node").textContent = id === data.root ? "root" : data.display(node?.prior_token);
  $("dist-strat").textContent =
    params.strategy === "topk" ? `top-k · k=${params.k}` :
    params.strategy === "topp" ? `top-p · p=${params.p.toFixed(2)}` :
    params.strategy === "beam" ? "beam · W=4" : "greedy · argmax";
  renderDistribution($("dist-bars"), $("dist-meta"), data, id, params);
}

function onTreeSelect(idStr, col) {
  const id = Number(idStr);
  path = path.slice(0, col).concat([id]);
  renderTree($("tree-svg"), data, path, onTreeSelect);
  renderDist();
  $("dice-out").textContent = "";
}

function onControlsChange() {
  renderDist();
}

function rollDice() {
  const step = sampleStep(data, selId(), params);
  const out = $("dice-out");
  if (!step) { out.textContent = "leaf node — no deeper tokens in the precomputed tree. Click a higher node to roll again."; return; }
  path = path.concat([step.childId]);
  renderTree($("tree-svg"), data, path, onTreeSelect);
  renderDist();
  out.innerHTML = `sampled <b>${escHtml(data.display(step.id))}</b> from the ${params.strategy} candidate set → descended. Roll again to keep walking.`;
}

function setupModes() {
  const ex = $("mode-explore"), ev = $("mode-evidence");
  const set = (m) => {
    mode = m;
    ex.classList.toggle("on", m === "explore");
    ev.classList.toggle("on", m === "evidence");
    $("explore-view").hidden = m !== "explore";
    $("evidence-panel").hidden = m !== "evidence";
    if (m === "evidence" && data) renderEvidence($("evidence-rows"), data);
  };
  ex.addEventListener("click", () => set("explore"));
  ev.addEventListener("click", () => set("evidence"));
}

init().catch((e) => {
  document.body.insertAdjacentHTML("beforeend", `<pre class="err">${e.message}\n${e.stack || ""}</pre>`);
  console.error(e);
});

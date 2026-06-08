// App wiring. Agent fleshes out across Tasks B1–B3. For now: load data + confirm the contract.
import { loadPromptFile } from "./data.js";

const DEFAULT_PROMPT = "sample";

async function init() {
  const data = await loadPromptFile(DEFAULT_PROMPT);
  document.getElementById("model-badge").textContent =
    `${data.meta.model}${data.meta.quantized ? " · " + data.meta.quantized : ""}`;

  // Scaffold sanity check — proves the data contract loads end-to-end.
  console.log("loaded prompt:", data.meta.prompt_text);
  console.log("root node:", data.node(data.root));
  console.log("greedy path:", data.greedyPath);

  // TODO(agent): renderTree / renderDistribution / renderControls / renderEvidence (Tasks B1–B3).
}

init().catch((e) => {
  document.body.insertAdjacentHTML("beforeend",
    `<pre style="color:#e8584a;padding:16px">${e.message}</pre>`);
  console.error(e);
});

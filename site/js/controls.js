// Strategy controls (right bottom): Greedy / Beam / Top-k / Top-p tabs + continuous sliders
// clamped to the slider caps (ADR 0001). onChange re-renders the distribution live; the 🎲 dice
// button is wired in main.js (it mutates tree state, which controls.js doesn't own).

const STRATS = [
  { id: "greedy", name: "Greedy" },
  { id: "beam", name: "Beam" },
  { id: "topk", name: "Top-k" },
  { id: "topp", name: "Top-p" },
];

export function renderControls(tabsEl, slidersEl, data, params, onChange) {
  tabsEl.innerHTML = STRATS.map(
    (s) => `<button class="tab ${s.id === params.strategy ? "on" : ""}" data-s="${s.id}">${s.name}</button>`
  ).join("");
  tabsEl.querySelectorAll(".tab").forEach((b) =>
    b.addEventListener("click", () => { params.strategy = b.dataset.s; onChange(); })
  );

  const tMax = data.tempMax, pMax = data.topPMax, kMax = data.topMSize;
  slidersEl.innerHTML =
    slider("temp", "temperature", 0.1, tMax, 0.05, params.temp, `0.1 – ${tMax} · capped so cutoffs stay within the stored top-M`) +
    (params.strategy === "topk" ? slider("k", "top-k", 1, kMax, 1, params.k, `1 – ${kMax} (stored top-M)`) : "") +
    (params.strategy === "topp" ? slider("p", "top-p", 0.05, pMax, 0.01, params.p, `0.05 – ${pMax}`) : "") +
    (params.strategy === "beam"
      ? `<div class="slider-cap">Beam width <b>W = 4</b>, precomputed to the 12-token horizon. The competing hypotheses live in the <b>Evidence</b> tab.</div>`
      : "");

  bind(slidersEl, "temp", (v) => (params.temp = v), onChange, false);
  if (params.strategy === "topk") bind(slidersEl, "k", (v) => (params.k = Math.round(v)), onChange, true);
  if (params.strategy === "topp") bind(slidersEl, "p", (v) => (params.p = v), onChange, false);
}

function slider(key, name, min, max, step, val, cap) {
  const shown = key === "k" ? String(Math.round(val)) : (+val).toFixed(2);
  return `<div class="slider" data-k="${key}">
    <div class="slider-head"><span class="name">${name}</span><span class="val" id="val-${key}">${shown}</span></div>
    <input type="range" min="${min}" max="${max}" step="${step}" value="${val}" />
    <div class="slider-cap">${cap}</div>
  </div>`;
}

function bind(root, key, set, onChange, isInt) {
  const wrap = root.querySelector(`.slider[data-k="${key}"]`);
  if (!wrap) return;
  const input = wrap.querySelector("input");
  const valEl = wrap.querySelector(`#val-${key}`);
  input.addEventListener("input", () => {
    const v = +input.value;
    set(v);
    valEl.textContent = isInt ? String(Math.round(v)) : v.toFixed(2);
    onChange();
  });
}

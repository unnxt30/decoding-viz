// Strategy controls. Agent implements in Task B3.
// Responsibility: strategy tabs (Greedy/Beam/Top-k/Top-p); continuous temp/p/k/beam-width
// sliders clamped to meta.slider_caps; 🎲 dice -> sample-walk one step (strategies.sampleFrom).

export function renderControls(tabsEl, slidersEl, diceEl, data, params, onChange) {
  // TODO(agent, Task B3): build tabs + sliders (clamped to data.meta.slider_caps), wire onChange.
}

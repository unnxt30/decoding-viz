// Browser-side decoding math — the JS MIRROR of precompute/decoding.py + distribution.py.
//
// DELIBERATELY NOT IMPLEMENTED YET. Writing these bodies would hand over the answer to the
// Python decoding exercise you're doing in learning mode (precompute/). Once you've implemented
// decoding.py / distribution.py, the agent mirrors them here (Task B2) so the sliders work live.
//
// Signatures are fixed so the rest of the frontend can be built against them.

/** softmax over an array of logits -> probabilities. */
export function softmax(logits) { throw new Error("not implemented (Task B2, after decoding.py)"); }

/** logits scaled by temperature t. */
export function applyTemperature(logits, t) { throw new Error("not implemented (Task B2)"); }

/** indices of the top-k logits. */
export function topKFilter(logits, k) { throw new Error("not implemented (Task B2)"); }

/** smallest index set whose cumulative prob >= p (the nucleus). */
export function topPFilter(logits, p) { throw new Error("not implemented (Task B2)"); }

/** true probabilities of a node's stored top-M, using its logsumexp (not subset-renormalized). */
export function trueProbs(node) { throw new Error("not implemented (Task B2)"); }

// Browser-side decoding math — the JS MIRROR of precompute/decoding.py + distribution.py.
//
// Now that the Python decoding core is implemented & verified, this mirrors it so the
// temperature / top-k / top-p sliders recompute live in the browser, with zero backend.
// Everything operates on a node's STORED distribution: top_m_tokens = [[id, logit], ...]
// (sorted desc by logit) + full-vocab logsumexp `lse` + residual `tail_mass`.

/** Stable softmax over an array of logits -> probabilities summing to 1 (subtract max). */
export function softmax(logits) {
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / z);
}

/** logits / t  (t<1 sharpens toward greedy, t>1 flattens toward uniform — claim #3). */
export function applyTemperature(logits, t) {
  return logits.map((l) => l / t);
}

/**
 * The node's distribution at temperature `t`, honest to the FULL vocab.
 *
 * At t=1:  p_i = exp(logit_i - lse)  — the TRUE probability (uses the stored full-vocab lse),
 *          and the stored `tail_mass` is the mass outside the top-M. Exact.
 * At t!=1: renormalize the top-M softmax to carry (1 - tail_mass), holding the tail fixed.
 *          Exact at t=1; a bounded approximation otherwise — the sliders are capped (top_p_max,
 *          temp_max) precisely so a cutoff never needs a token past the stored top-M (ADR 0001).
 *
 * Returns { dist: [{id, logit, p}], tail } with dist sorted descending by p.
 */
export function nodeDistribution(node, t = 1) {
  const pairs = node.top_m_tokens;                 // [[id, logit], ...] already sorted desc
  const scaled = pairs.map(([, l]) => l / t);
  const sm = softmax(scaled);                      // sums to 1 across the top-M
  const head = 1 - (node.tail_mass ?? 0);          // mass the top-M should carry
  const dist = pairs.map(([id, logit], i) => ({ id, logit, p: sm[i] * head }));
  return { dist, tail: node.tail_mass ?? 0 };
}

/** Set of the top-k ids (rigid — always exactly k). `dist` is already sorted desc. */
export function topKSet(dist, k) {
  return new Set(dist.slice(0, Math.max(0, k)).map((d) => d.id));
}

/**
 * Nucleus set: the smallest prefix whose cumulative probability >= p (the crossing token is
 * INCLUDED, per the nucleus-sampling paper). Adaptive — shrinks when confident, grows when flat.
 */
export function topPSet(dist, p) {
  const keep = new Set();
  let cum = 0;
  for (const d of dist) {
    keep.add(d.id);
    cum += d.p;
    if (cum >= p) break;
  }
  return keep;
}

/** Sample one id from a candidate set, renormalized over just those members (the 🎲 step). */
export function sampleFrom(dist, candidateIds) {
  const cands = dist.filter((d) => candidateIds.has(d.id));
  const z = cands.reduce((a, d) => a + d.p, 0);
  let r = Math.random() * z;
  for (const d of cands) {
    r -= d.p;
    if (r <= 0) return d.id;
  }
  return cands.length ? cands[cands.length - 1].id : null;
}

/** Greedy pick = argmax (deterministic). Equivalent to topKSet(dist, 1). */
export function greedyId(dist) {
  return dist.length ? dist[0].id : null;
}

/**
 * Resolve the candidate set for a strategy at given params.
 * strategy: "greedy" | "topk" | "topp"  (beam is path-based, handled elsewhere).
 * Returns { candidates: Set<id>|null, kept }. null candidates => no cut (greedy shows all).
 */
export function candidateSet(dist, strategy, { k, p }) {
  if (strategy === "topk") { const s = topKSet(dist, k); return { candidates: s, kept: s.size }; }
  if (strategy === "topp") { const s = topPSet(dist, p); return { candidates: s, kept: s.size }; }
  return { candidates: null, kept: dist.length };
}

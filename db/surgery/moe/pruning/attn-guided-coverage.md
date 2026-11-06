# Attribution-Guided, Coverage-Maximized pruning (structural MoE)

Source: Ding et al., "Attribution-Guided and Coverage-Maximized Pruning for
Structural MoE Compression", arXiv 2606.18304 — ICML 2026 SPOTLIGHT.

Key observation: information inside an expert is concentrated in a FEW channels, so
even "high-importance" experts hold large redundancy. Pruning whole experts throws
away the live channels with the dead ones.

Method: treat pruning as a COVERAGE problem over channel-attribution scores — keep a
minimal set of channels that covers the model's behaviour — solved with an efficient
attribution-based approximation. Structural, kernel-compatible with low-bit quant.

Results (DIRECTLY ON OUR MODEL): on Qwen3-30B-A3B, 5.27x memory reduction when
combined with 4-bit quant, accuracy preserved at 25% / 50% structured pruning.

Relevance for us (TOP PICK): the only method validated on the exact base we are
using, and it keeps per-expert live channels instead of deleting whole experts —
lower risk to the spread-out tool-calling capability. Pair with 4-bit + SLAB stream.

Related: `../scoring/reap.md`, `hope.md`, `maestro.md`, `../healing/router-kd.md`.

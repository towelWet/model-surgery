# NAEE / DiEP / EvoESAP — non-uniform + skip pruning family (compact)

Three related ideas; grouped because each is a variant on "do not prune uniformly".

NAEE — "Not All Experts are Equal", arXiv 2402.14800 (2024). Post-training expert
DROPPING + SKIPPING guided by router/expert signals; documents large expert
imbalance. The baseline EEP and others beat.

DiEP — arXiv 2509.16105 (2025). DIFFERENTIABLE expert pruning with NON-UNIFORM
per-layer rates; argues uniform sparsity across layers is suboptimal.

EvoESAP — arXiv 2603.06003 (2026). Non-uniform per-layer expert ALLOCATION on top of
a pruning criterion; pick how many experts to keep per layer, not a flat ratio.

Relevance for us: all say the same actionable thing — allocate the keep-budget
per-layer, not flat. Early/late layers tolerate different cuts. Layer when we prune.

Related: `eep.md`, `attn-guided-coverage.md`.

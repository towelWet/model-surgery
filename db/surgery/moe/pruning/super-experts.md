# Super Experts — never prune the load-bearing few

Source: "Unveiling Super Experts in MoE LLMs", arXiv 2507.23279.

A small number of experts are disproportionately critical: removing them collapses
model quality far more than their saliency score alone suggests. Compression built
around Super-Expert PRESERVATION is an active 2026 direction.

Rule for any surgery here: before pruning/compressing, identify the super experts
(highest saliency AND highest co-activation centrality) and put them in a protection
set that is kept full-precision regardless of the global budget.

Ties into LorExperts' "dominant" selection and HOPE's pairwise view.

Related: `db/surgery/moe/compression/lorexperts.md`, `hope.md`.

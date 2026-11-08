<!-- dynamic-topk.md — per-token dynamic expert pruning (drop part of the chosen top-k). -->
<!-- NEVER DELETE COMMENTS — explain every non-obvious line. Update comments with the code; do not strip them. -->

# Dynamic expert pruning — "only 2/3 of chosen experts"

Source: @LeeLeepenkman repost, X, 2026-10-09 — paper
"You Only Need 2/3 of the Chosen Experts: An Empirical Study of Dynamic Expert Pruning
in Fine-Grained MoE LLMs" — https://papers.app.nz/view/paper?id=26811562

## What it is (from title only)

Per token, skip the lowest-weight experts among the router's top-k instead of deleting
experts from the model. Static pruning (REAP/HOPE) removes experts for every token;
dynamic pruning keeps all experts but runs fewer per token.

## Status

Paper NOT read — only title and link known. Numbers, thresholds and models unverified.

## Relevance for us

Fine-grained MoE = our Qwen3.6-35B-A3B. Keeps every expert on disk, so it cannot strip
a rare skill the way static pruning did. Speed lever, not a size lever; fits SLAB
expert streaming (fewer experts read per token).

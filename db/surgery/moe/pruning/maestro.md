# MAESTRO — current SOTA one-shot expert pruning

Source: "It Takes a MAESTRO To Prune Bad Experts", arXiv 2607.08601 (2026).

Claim: beats the best prior baseline (REAP) by ~1.26% at 25% compression and ~3.60%
at 50% on GPT-OSS-20B. Positions itself as the new one-shot SOTA over REAP.

Relevance for us: a drop-in better scorer/pruner than REAP if we go the
whole-expert-removal route. Not yet validated on Qwen3-30B-A3B in the sources seen,
so treat the margin as model-transfer-uncertain. Secondary to
`attn-guided-coverage.md` (which IS validated on our model).

Related: `../scoring/reap.md`, `attn-guided-coverage.md`, `hope.md`.

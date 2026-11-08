<!-- razor.md — RAZOR: pruning replaceable experts (paper pointer; body NOT read yet). -->
<!-- NEVER DELETE COMMENTS — explain every non-obvious line. Update comments with the code; do not strip them. -->

# RAZOR — Pruning Replaceable Experts in LLMs

Source: @Memoirs, X, 2026-09-29 — https://x.com/Memoirs/status/2104900421529542979

Paper: Mingyang Song, Mao Zheng — https://arxiv.org/abs/2609.30465 (cs.LG, cs.CL)

## What we know

Only title + authors from repost. Title says idea: drop experts whose job other
experts can take over ("replaceable"), not just low-use ones.

## Not yet known

Scoring rule, calibration needs, results, code. Read paper before citing any number.

## Relevance for us

Possible third scorer beside REAP (`../scoring/reap.md`) and HOPE (`hope.md`):
redundancy-based instead of saliency-based. Untested here.

Correction 2026-11-02: older INDEX called RAZOR "not a technique". Wrong in general —
it is a real paper. Still true that nothing in this stack implements it.

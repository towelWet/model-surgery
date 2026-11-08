# AIMER — calibration-free task-agnostic expert pruning

Source: arXiv 2603.18492.

Scores each expert from WEIGHTS ONLY — no calibration corpus, no activations:

    score = mean(|W|) / RMS(W)     # per expert

One-shot. Expert scoring is ~0.22–1.27 s vs REAP's 0.75–2.96 HOURS (~3000x faster),
because it never runs the model. Calibration-free, so it sidesteps the proxy-calibration
damage that wrecked our REAP runs (see `../scoring/reap.md`, `~/.claude/skills/ai-customization/surgery/80e-defect.md`).

Reported retention at aggressive pruning: Qwen3-30B @50% prune → coding 36.1% vs
REAP 4.6%. Wins coding/agentic + aggressive ratios; REAP can edge it on math/creative.

Our result (2026-10, Qwen3.6-35B-A3B — see
`~/.claude/skills/ai-customization/surgery/pruning-results-2026-10.md`):
built 128e (not trialed alone), 96e → garbage tool args (t72), 64e → garbled (75% prune).
Same failure CLASS as REAP: one-shot pruning at 96e and below broke god's tool-arg
binding. AIMER did not rescue tool-calling; the winning path dropped pruning entirely for
intact 256e SSD streaming (SLAB — see `../INDEX.toml` [serving]).

Related: `db/surgery/moe/scoring/reap.md` (AIMER is the calibration-free alternative), `hope.md` (second-order
successor), Recover-LoRA (data-free heal if a pruned model under-retains).

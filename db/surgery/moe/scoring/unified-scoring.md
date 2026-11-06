# Unified one-shot scoring principle

Source: "How to Score Experts for One-Shot MoE Expert Pruning", arXiv 2606.15716 (2026).

All one-shot expert scores are products drawn from three signals:

    score ~ f( routing_frequency , gate_mass , activation_norm )

Key finding: no single member of this family wins everywhere.
- TASK-SPECIFIC pruning (our RE goal) benefits from keeping frequency + gate signals.
- TASK-AGNOSTIC pruning favours gate-free norms.

Implication for us: since the target is a specific domain (macOS RE / tool-calling),
keep all three — frequency × gate × norm. Our REAP `g·‖e‖` has gate + norm; add a
routing-frequency weight when scoring on the RE calibration set.

Related: `db/surgery/moe/scoring/reap.md`, `db/surgery/moe/pruning/domain-coding-prune.md`.

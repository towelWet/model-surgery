# MoE-Pruner expert-wise KD — heal by distilling from the intact teacher

Source: "MoE-Pruner: Pruning Mixture-of-Experts", arXiv 2410.12013 (2024).

Method: prune, then distill EXPERT-WISE from the unpruned model as teacher — each
surviving expert is trained to reproduce the teacher's per-expert outputs.

Result claimed: pruned model keeps 99% of original performance AFTER the expert-wise
KD step (Mixtral teacher).

Relevance for us: a stronger (but heavier) heal than router-only KD — it touches the
experts, not just the router. Use when Router KD alone under-recovers tool-calling.
Order: prune -> Router KD (cheap) -> if still short, expert-wise KD.

Related: `router-kd.md`, `lora-self-distill.md`.

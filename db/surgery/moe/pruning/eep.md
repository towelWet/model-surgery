# EEP — Efficient Expert Pruning (gradient-free evolutionary)

Source: "Efficient Expert Pruning for Sparse MoE LLMs", arXiv 2407.00945 (Jul 2024).

Method: gradient-free EVOLUTIONARY search over which experts to prune; uses expert
MERGING as a memory-efficient stand-in for fine-tuning the pruned model. No
backprop needed.

Result claimed: prune up to 75% of experts in Mixtral 8x7B-Instruct with minimal
loss; beats NAEE / random / frequency / soft-activation baselines at 30 experts.

Relevance for us: evolutionary search could directly optimize the keep-set AGAINST a
tool-calling objective (our real failure metric) rather than a generic corpus — a
way to stop calibration bias deleting the agentic experts. Heavier (many eval passes).

Related: `db/surgery/moe/pruning/naee-diep-evoesap.md`, `../merging/hc-smoe.md`.

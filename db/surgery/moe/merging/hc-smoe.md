# HC-SMoE — hierarchical-clustering expert merging (retraining-free)

Source: Chen et al., "Retraining-Free Merging of Sparse MoE via Hierarchical
Clustering", arXiv 2410.08589.

Method: cluster experts by AVERAGE ACTIVATION OUTPUT over calibration tokens
(Euclidean, agglomerative), then merge each cluster by frequency-weighted parameter
averaging. Task-agnostic, routing decisions do NOT drive the merge (uses outputs).

Result context: strong on multiple-choice QA without fine-tuning — BUT the REAP
paper reports HC-SMoE collapses to 65.2% of baseline code-gen at 50% on
Qwen3-30B-A3B, vs REAP's 95.9%.

Relevance for us: merging LOSES to pruning on our exact model for generative/code
work. Keep as reference; do NOT choose merging as the primary path. If merged, heal
with Router KD (see `../healing/router-kd.md`, which cites merging as a paradigm).

Related: `m-smoe.md`, `../scoring/reap.md`.

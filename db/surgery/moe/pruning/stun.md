# STUN — Structured-Then-UNstructured pruning

Source: Lee et al., "STUN: Structured-Then-Unstructured Pruning for Scalable MoE
Pruning", ACL 2025.

Method: stage 1 = expert-level structured pruning driven by expert BEHAVIOUR
SIMILARITY, needing only a SINGLE forward pass (vs O(kn/sqrt n) passes elsewhere);
stage 2 = unstructured pruning on the expert-pruned model, which tolerates it well.

Result: on Snowflake Arctic (480B, 128 experts) near-zero loss at 40% sparsity, even
on generative GSM8K.

Relevance for us: the cheapest scorer to run (one forward pass), good for a fast
first pass to rank experts by behaviour similarity before a heavier method. Same
128-expert granularity as our base.

Related: `../scoring/reap.md`, `attn-guided-coverage.md`.

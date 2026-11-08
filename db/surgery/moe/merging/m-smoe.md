# M-SMoE — routing-signal expert merging

Source: Li et al., 2023 (SMoE merge-compress line; cited by HC-SMoE and REAP).

Method: seed cluster centres from globally DOMINANT experts (usage frequency across
layers); assign the rest by COSINE SIMILARITY of router logits; align weights with
weight-matching; merge by frequency-weighted averaging.

Difference from HC-SMoE: M-SMoE uses ROUTING signals (router logits + usage freq);
HC-SMoE uses output-activation similarity. HC-SMoE generally reported stronger.

Relevance for us: same verdict as HC-SMoE — merging underperforms pruning on
30B-A3B generative tasks. Reference only.

Related: `hc-smoe.md`.

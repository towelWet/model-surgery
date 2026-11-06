# HOPE — higher-order pruning of experts

Source: lacuna.tiptreesystems.com / 2026.
Paper: "Higher-order pruning of experts in mixture-of-experts language models",
Tseng, Kaul, Zancato, Xia, Soatto (AWS Agentic AI) — https://arxiv.org/abs/2609.18916
(via @Memoirs / @fly51fly X reposts, 2026-09).

Standard pruning scores each expert independently. HOPE accounts for the COOPERATIVE,
pairwise interactions between experts — two experts that are individually low-saliency
may be jointly load-bearing (or jointly redundant).

Claim: up to 50% expert removal with minimal performance degradation, by modelling
pairwise expert cooperation rather than per-expert scores alone.

Relevance for us: a refinement on top of REAP/domain scoring — when deciding which
experts to drop, check pairwise co-contribution so we don't remove a pair that
together carries RE behaviour. Secondary to `db/surgery/moe/compression/lorexperts.md`; useful if we prune.

Related: `db/surgery/moe/scoring/reap.md`, `super-experts.md`.

# ConMoE — expert-pool consolidation via prototype reassignment

Source: arXiv 2605.29350 (2026).

Formulates post-training MoE compression as CONSOLIDATION, not deletion: keep a
smaller set of pretrained experts as reusable PROTOTYPES, with a deterministic
reassignment map from every original expert to its chosen prototype.

Contrast with LorExperts: ConMoE reassigns (many→one prototype, no per-member
correction); LorExperts keeps a dominant + low-rank corrections per member. ConMoE is
lighter but coarser.

Relevance: a cheaper alternative if LorExperts' SVD refit is too heavy and
domain-prune drops too much — consolidate redundant experts onto prototypes while
keeping the RE prototypes.

Related: `db/surgery/moe/compression/lorexperts.md`, `~/.claude/skills/ai-customization/surgery/crackme-surgery-plan.md`.

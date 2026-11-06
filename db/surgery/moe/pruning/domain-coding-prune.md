# Domain-calibrated pruning — "Half the Experts, All the Code"

Source: arXiv 2607.16721 (2026). One-shot DOMAIN pruning of MoE for CODING — our
exact scenario (RE/code is a code domain).

Method:
- Build a CODE/domain calibration set (representative code/RE samples, not general text).
- Score each expert by loss/Taylor sensitivity: how much removing it increases loss on
  that domain calibration set.
- Router PRESERVED (unchanged). Keep ~50% of experts per layer; counts vary by layer.

Result: "frontier reversal" — domain pruning beats corpus-agnostic pruning at EQUAL
model size on code tasks; general pruning degrades significantly on code.

For us: calibrate on RE/tool-call demos, keep the RE experts, prune the rest via
`tensor-array-prune-experts` (already slices expert tensors + router rows + config +
shards consistently). This is the fallback if LorExperts low-rank refit is too heavy.

Related: `db/surgery/moe/compression/lorexperts.md` (primary), `db/surgery/moe/scoring/unified-scoring.md`, `~/.claude/skills/ai-customization/surgery/crackme-surgery-plan.md`.

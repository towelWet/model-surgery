# LorExperts / BTExperts — router-preserving low-rank compression (PRIMARY)

Source: AMD, arXiv 2608.07814 (2026). Does NOT delete experts → best fit for keeping
rare specialists (our failure mode). MMLU 54.9 vs 45.7 for prior delta-decomposition.

Procedure:
1. Cluster E experts into k≈E/2 via k-medoids on CO-ACTIVATION distance (NPMI of which
   experts fire together), not weight similarity. Seed medoids from highest-firing
   experts (routing is concentrated: ~64 of 128 absorb ~94.5% of tokens).
2. Each cluster keeps its highest-firing DOMINANT expert at full precision.
3. Each other member m: align neurons to dominant (Hungarian / Jonker–Volgenant on
   concatenated gate/up/down vectors), then truncated SVD of the residual
   R = W_m^P − W_dom ≈ U_r Σ_r V_rᵀ ; store B_m=U_rΣ_r^½, A_m=Σ_r^½V_rᵀ ;
   W_m ≈ W_dom + B_m A_m. rank r≈64 → ~½ expert params.
4. Optional reconstruction refit: min ‖(W_dom+B_mA_m)x − W_m x‖² on cached calib
   activations — one closed-form SVD solve, ~100× cheaper than gradient steps.

Why it preserves capability: all experts + router stay addressable; domain specialists
live in rare co-activation communities kept via the dominant + corrections. BTExperts
= inference-only binary-tree sharing of dominants, ~1.9× bandwidth speedup.

Related: `db/surgery/moe/pruning/domain-coding-prune.md` (fallback), `~/.claude/skills/ai-customization/surgery/crackme-surgery-plan.md`.

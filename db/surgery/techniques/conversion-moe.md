# Technique families — MoE conversion & compression

Post-hoc restructuring without full retrain. MoE is one subset.

## 1. Dense → MoE (MoEfication / Upcycling)

Partition FFN into experts; train routers only.

**Neuron partition (Whittle):** `E × S + H = W` (exact — no invented FFN weights).
Qwen3.8-27B example: `64 × 192 + 5120 = 17408`.

**Upcycling:** copy MLP E times → identical experts + learned router.
MistralMoE, early DeepSeek-MoE, Drop-Upcycling.

**ToMoE:** dense → sparse MoE via learned token routing, fixed per-token budget.
[arxiv:2501.15316](https://arxiv.org/abs/2501.15316)

**ExpertWeaver:** latent MoE from GLU activation patterns.
[arxiv:2602.15521](https://arxiv.org/pdf/2602.15521)

**DOT-MoE:** differentiable optimal transport neuron → expert assignment.
[arxiv:2606.01666](https://arxiv.org/pdf/2606.01666)

## 2. MoE compression (prune / merge)

**Expert pruning** — usage, routing mass, activations, importance.
[MoE-Pruner](https://arxiv.org/pdf/2410.12013)

**Expert merging** — routing stats, output similarity, clustering, subspace fusion.

**ConMoE** — remap experts to smaller prototype pool.
[arxiv:2605.29350](https://arxiv.org/pdf/2605.29350)

**MoE → Dense distillation** — compress sparse MoE to small dense.
[arxiv:2605.28207v1](https://arxiv.org/html/2605.28207v1)

Expert redundancy is **heterogeneous across layers** — uniform budgets are suboptimal.

Toolkit: `pipeline/surgery/moefy/entry.py`, `pipeline/surgery/expert/prune.py <src> <dest-Tau> <usage_json>`,
`pipeline/profile/moe_fidelity/run.py`. `pipeline/profile/expert/usage.py` is still a placeholder
(needs tensor-array `/v1/internals/router_logits`).

## Hybrid: packed dense + affine MoE (Bonsai)

Dense linears: 1-bit/ternary **implemented** via `qmm_dispatch` (host GEMM + Metal JIT).
MoE experts stay affine 2–8 bit (qwen3 moe, deepseek v2, 4_exp 4/8).
Packed `gather_qmm` **not needed** — experts are not bits=1/ternary.
`qmm_use_metal()` default false; enable with `TARR_USE_METAL=1`.

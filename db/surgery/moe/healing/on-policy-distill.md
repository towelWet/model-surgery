<!-- on-policy-distill.md — on-policy (self-)distillation as heal: student learns from its OWN outputs. -->
<!-- NEVER DELETE COMMENTS — explain every non-obvious line. Update comments with the code; do not strip them. -->

# On-policy distillation — heal on student's own generations

## Sources (reposts; papers NOT read yet)

| Who | Date | Claim |
|---|---|---|
| @dair_ai on NVIDIA Nemotron 3.5 | 2026-10-30 | pipeline: SFT → RL → multi-teacher on-policy distillation |
| @jiqizhixin on d-OPSD | 2026-07-17 | on-policy self-distill for diffusion LLMs; beats RLVR + SFT at ~10% steps; arxiv.org/pdf/2606.18195 |
| @jayden_teoh_ Soft-MTP | 2026-10-30 | KL self-distill grad into next-token head for free: `h_ntp += h_mtp.detach() - h_mtp`; arxiv 2610.38331 |

## Idea

Off-policy KD (ours today: `cache_teacher_logits.py`) trains on fixed text.
On-policy: student generates, teacher scores THOSE tokens, student fixes its own
mistakes. Errors a pruned model actually makes (garbage tool args) get supervised.

## Relevance for us

Fits after router KD (`router-kd.md`) if tool-arg errors remain: let pruned model
run god RE traces, teacher = intact SLAB, distill on student's turns.
Costs teacher forward per student sample — heavier than router KD.
d-OPSD is diffusion-LLM specific; take only the on-policy idea.
Soft-MTP matters only if we train an MTP head.

# Technique families — prune, distill, active-param speed

## 3. Structural pruning (dense)

Remove neurons, heads, or layers. LLM-Pruner (gradient), FLAP (output stability).
Irreversible capacity loss; sharp degradation at high ratios — pair with fine-tune.

```bash
python3 surgery/depth_prune.py <src> <dest> "<spec>"
python3 surgery/mlp_only_prune.py <src> <dest-Tau> "<spec>"
python3 surgery/width/prune.py <src> <dest-Tau> <new_interm>
```

Root: `./`. Dest names: `db/surgery/methodology/discipline.md`.

## Metal JIT 1-bit (no prune)

Weight-size cut is **format**, not surgery. `bits=1` = 32× density vs F16
(actual **4.2 GB** vs **~54 GB** bf16). No `depth_prune` / `mlp_only_prune` / `width_prune`.
**Implemented** (host GEMM + Metal JIT wired). Serve Bonsai via `TARR_USE_METAL=1`. Roofline annex.

## 4. Knowledge distillation

Small student mimics teacher **logits** (not labels only). Often after cuts above.
Whittle uses top-32 logit distillation from parent 27B.
Paths: `heal/cache_teacher_logits.py`, local-only history: db/rejected/mlx-lm-heal/train_lora_distill.py.

## Speed: active param ratio

Inference cost ∝ **active** params, not total.
Decode (memory-bound): `tok/s_max ≈ BW / bytes_per_token`.

| Model | Active | Total | Ratio | Verdict |
|---|---|---|---|---|
| Qwen3-30B-A3B | 3B | 30B | 0.10 | real win |
| Whittle-Next-27B-A3B | ~3B | 27B | ~0.11 | real win |
| Whittle-MoE-27B | 17.8B | 27B | 0.66 | marginal |
| Dense 27B | 27B | 27B | 1.00 | baseline |

**Rule:** active/total > 0.5 → barely worth calling MoE for speed.

Roofline — Bonsai-27B-mlx-1bit (**implemented**; tok/s theoretical, not loaded tonight):
**120 GB/s / ~4.2 GB ≈ ~28 tok/s** max.

Site probes: optional private speed diagnosis, `pipeline/profile/fast/bench/core.py`.
Active-B examples: SKILL_TECHNIQUES_PRUNE_ANNEX.md [missing: not on disk 2026-11-06].
Two-level async: `async_eval` per token overlaps decode; `eval_every=8` at layers cuts intra-forward sync.

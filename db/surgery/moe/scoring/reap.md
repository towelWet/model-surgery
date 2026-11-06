# REAP — router-weighted expert activation pruning

Source: Cerebras / U.Calgary, arXiv 2510.13999 (Oct 2025, still the baseline).

Saliency per (layer, expert):

    c = g · ‖e‖        # router gate value × L2 norm of expert output

One-shot, no retraining. Rank experts per layer, remove lowest-saliency to a target
ratio. Reported <8% generative drop at >50% pruning; beats merging (removing an
expert entirely preserves the router's input-conditioned control better than
blending).

Our code already implements this: `tensor-array/src/arch/qwen/3_5/moe/reap_stats.cpp`
computes `g·‖e‖`. Math is correct; what was wrong was the CALIBRATION corpus (proxy,
general) — see `~/.claude/skills/ai-customization/surgery/80e-defect.md` and `db/surgery/moe/pruning/domain-coding-prune.md`.

Ref model (pre-REAP'd 35B-A3B): huggingface.co/cerebras/Kimi-Linear-REAP-35B-A3B-Instruct

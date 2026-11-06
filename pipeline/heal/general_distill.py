# general_distill.py — general corpus SFT via tensor-array-train (no mlx_lm in heal path).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Run AFTER tool_distill.py; lower LR to avoid overwriting tool routing gains.
# Optional --cache-first archives teacher top-32 NPZ via cache_teacher_logits.py.
# Usage: python3 general_distill.py --student PATH --data JSONL
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "distill"))
from _cache_first import cache_first
from _distill_args import distill_args
from _train import train

CACHE_DEFAULT = os.environ.get("SURGERY_TEACHER_CACHE", "heal/cache/kd_top32.npz")

args = distill_args(iters=200, lr=5e-6, adapter_path="heal/adapters/Tau-general-heal")  # gentler than tool pass
cache = args.teacher_cache or CACHE_DEFAULT
if args.cache_first:
    if not args.teacher:
        print("--cache-first requires --teacher PATH")
        sys.exit(1)
    cache_first(args, cache)
train(args, cache)

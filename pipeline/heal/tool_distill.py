# tool_distill.py — tool-routing heal via cached teacher top-32 + tensor-array-train SFT.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Step 1: cache_teacher_logits.py on tool-heavy JSONL (teacher on tensor-array only).
# Step 2: tensor-array-train LoRA on student with optional --teacher-cache (CE + top-32 KL).
#
# Usage:
#   python3 tool_distill.py --student PATH --data JSONL [--teacher-cache NPZ]
#   python3 tool_distill.py --student PATH --data JSONL --cache-first --teacher PATH
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "distill"))
from _cache_first import cache_first
from _distill_args import distill_args
from _train import train

CACHE_DEFAULT = os.environ.get("SURGERY_TEACHER_CACHE", "heal/cache/kd_top32.npz")

args = distill_args(iters=500, lr=1e-5, adapter_path="heal/adapters/Tau-tool-heal")
cache = args.teacher_cache or CACHE_DEFAULT
if args.cache_first:
    cache_first(args, cache)
train(args, cache)
print("Next: fuse_adapter.py --base STUDENT --adapter ADAPTER --out ...-Tau-...")

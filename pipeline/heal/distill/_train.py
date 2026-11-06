# _train.py — tensor-array-train LoRA SFT on student; adds top-32 KL term when teacher cache exists.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # heal/ helpers
from _train_bin import train_bin                          # env/PATH tensor-array-train (never mlx_lm)


def train(args, cache):
    os.makedirs(args.adapter_path, exist_ok=True)
    cmd = [
        train_bin(), "train",
        "--model", args.student,
        "--data", args.data,
        "--out", args.adapter_path,
        "--rank", str(args.rank),
        "--alpha", str(args.rank * 2),
        "--steps", str(args.iters),
        "--max-seq", "2048",
        "--lr", str(args.lr),
    ]
    if os.path.exists(cache):
        cmd += ["--teacher-cache", cache]                   # CE + top-32 KL
    else:
        print(f"No teacher cache at {cache}; CE-only SFT. Run cache_teacher_logits.py first.")
    print("Train:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nAdapter -> {args.adapter_path}")

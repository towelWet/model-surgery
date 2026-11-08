# _args.py — shared CLI for tool/distill.py / general_distill.py; only defaults differ.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse


def distill_args(iters, lr, adapter_path):
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--teacher", help="Teacher path for --cache-first serve step")
    p.add_argument("--teacher-cache", default=None, help="NPZ from cache_teacher_logits.py")
    p.add_argument("--cache-first", action="store_true", help="Run cache_teacher_logits before train")
    p.add_argument("--restore", default=None, help="Model to swap back to after caching")
    p.add_argument("--iters", type=int, default=iters)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--rank", type=int, default=16)          # LoRA rank; alpha = 2 * rank
    p.add_argument("--adapter-path", default=adapter_path)
    return p.parse_args()

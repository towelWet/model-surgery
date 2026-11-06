# cache_args.py — CLI flags for heal/cache_teacher_logits.py.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse


def cache_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="distill_corpus.jsonl")
    p.add_argument("--out", default=None, help="Output .npz (default: cache_root/kd_top32.npz)")
    p.add_argument("--teacher", help="Teacher MLX path; with --serve runs swap_model.sh first")
    p.add_argument("--serve", action="store_true", help="Unload resident server and load --teacher")
    p.add_argument("--restore", help="After cache, swap back to this model (e.g. Tau-56L)")
    return p.parse_args()

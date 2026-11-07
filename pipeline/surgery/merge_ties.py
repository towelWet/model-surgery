# merge_ties.py — legacy filename for DARE single-task-vector arithmetic (not TIES).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage:
#   python3 merge_ties.py <src_a> <src_b> <dest> [--density 0.5] [--lambda 1.0]
#                        [--weight-b 1.0] [--seed 42] [--bf16-only]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _merge_ties import run_merge
from checkpoint.destination import validate_destination


def main():
    p = argparse.ArgumentParser(description="DARE single-task-vector merge; TIES is not implemented")
    p.add_argument("src_a", help="base checkpoint (task-vector origin)")
    p.add_argument("src_b", help="second checkpoint")
    p.add_argument("dest", help="output dir (must not exist)")
    p.add_argument("--density", type=float, default=0.5, help="DARE keep probability p")
    p.add_argument("--lambda", dest="lam", type=float, default=1.0, help="global task-vector scale")
    p.add_argument("--weight-b", type=float, default=1.0, help="weight on src_b task vector")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16-only", action="store_true", help="skip requant; write bf16 weights")
    args = p.parse_args()

    validate_destination(args.dest, sources=[args.src_a, args.src_b])  # before any writes

    run_merge(
        args.src_a,
        args.src_b,
        args.dest,
        args.density,
        args.lam,
        args.weight_b,
        args.seed,
        args.bf16_only,
    )


if __name__ == "__main__":
    main()

# entry.py — CLI: low-rank (SVD / activation-aware ASVD) compression of decoder linears.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <bf16_src> <dest> --ratio 0.5 [--roles gate,up,down] [--calib DIR --alpha 0.5] [--factored]
#   --ratio = parameters kept per matrix; --calib (capture.py taps) switches plain SVD to ASVD.
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.lowrank._run import run_lowrank

mx.set_default_device(mx.cpu)                               # host SVD; GPU is shared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--roles", default="q,k,v,o,gate,up,down")
    p.add_argument("--calib", default=None)
    p.add_argument("--alpha", type=float, default=0.5)      # ASVD activation exponent (paper: 0.5)
    p.add_argument("--factored", action="store_true")        # store A, B in shards (runtime serves them)
    a = p.parse_args()
    rep = run_lowrank(a.src, a.dest, a.ratio, set(a.roles.split(",")), a.calib, a.alpha, a.factored)
    print(f"factored {len(rep)} tensors; mean rel_err {sum(r['rel_err'] for r in rep) / max(1, len(rep)):.4f}: {a.dest}")


if __name__ == "__main__":
    main()

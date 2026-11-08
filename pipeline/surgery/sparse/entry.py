# entry.py — CLI: unstructured pruning (zero individual weights) of a bf16 checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <bf16_src> <dest> --sparsity 0.5 [--nm 2:4] [--calib DIR] [--method magnitude|wanda|sparsegpt]
#   --calib: profile/moefy/capture.py output -> Wanda scores on MLP; without it, magnitude.
#   --nm N:M: semi-structured pattern (N kept per M inputs); overrides --sparsity.
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.sparse._run_sparse import run_sparse
import mlx.core as mx

mx.set_default_device(mx.cpu)                             # host math; GPU shared with other apps hit ~2s watchdog on big dequant/save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--sparsity", type=float, default=0.5)  # fraction of weights zeroed per row
    p.add_argument("--nm", default=None)                   # e.g. "2:4"
    p.add_argument("--calib", default=None)                # capture.py taps (attn/o/mlp) for Wanda
    p.add_argument("--roles", default="q,k,v,o,gate,up,down")  # e.g. gate,up,down = MLP only
    p.add_argument("--method", choices=("magnitude", "wanda", "sparsegpt"), default=None)  # default: wanda if --calib
    a = p.parse_args()
    if not 0.0 < a.sparsity < 1.0:
        sys.exit("--sparsity must be in (0, 1)")
    nm = tuple(int(v) for v in a.nm.split(":")) if a.nm else None
    rows = run_sparse(a.src, a.dest, a.sparsity, nm, a.calib, set(a.roles.split(",")), a.method)
    print(f"pruned {len(rows)} tensors; mean zero fraction {sum(r['zero'] for r in rows) / len(rows):.3f}")


if __name__ == "__main__":
    main()

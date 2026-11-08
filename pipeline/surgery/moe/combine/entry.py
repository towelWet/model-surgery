# ../moe/combine/entry.py — CLI: combine dense fine-tunes into one MoE (one expert per fine-tune).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <quantized_base> <dest> --experts FT1 FT2 ... --calib C1 C2 ... [--topk 1]
#   Ci = profile/moefy/capture.py --texts output of expert i's own prompts, captured on base.
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.moe.combine._run import combine

mx.set_default_device(mx.cpu)                               # host work; GPU shared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("dest")
    p.add_argument("--experts", nargs="+", required=True)
    p.add_argument("--calib", nargs="+", required=True)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--logit-std", type=float, default=2.0)
    a = p.parse_args()
    combine(a.base, a.experts, a.calib, a.dest, a.topk, a.logit_std)
    print(f"combined {len(a.experts)} experts (top-{a.topk}): {a.dest}")


if __name__ == "__main__":
    main()

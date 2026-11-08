# ../heads/linear/entry.py — CLI: linear-attention (Gated DeltaNet) head pruning for Qwen3.5/3.6 hybrids.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <bf16_src> <dest> --drop 4 --calib DIR
#   DIR = profile/moefy/capture.py output (needs "o.<i>" for linear layers).
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.heads.linear._run import run_heads_linear

mx.set_default_device(mx.cpu)                               # host work; shared GPU hit watchdog before


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--drop", type=int, required=True)       # key-head groups removed per linear layer
    p.add_argument("--calib", required=True)
    a = p.parse_args()
    run_heads_linear(a.src, a.dest, a.drop, a.calib)
    print(f"dropped {a.drop} linear key-head groups per layer -> {a.dest}")


if __name__ == "__main__":
    main()

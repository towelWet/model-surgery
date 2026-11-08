# entry.py — CLI: attention-head pruning (drop least important KV groups per layer).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <bf16_src> <dest> --drop 2 --calib DIR
#   DIR = profile/moefy/capture.py output (needs "o.<i>" per-head outputs).
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.heads._run import run_heads

mx.set_default_device(mx.cpu)                               # host work; shared GPU hit watchdog before


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--drop", type=int, required=True)       # KV groups removed from every layer
    p.add_argument("--calib", required=True)
    a = p.parse_args()
    run_heads(a.src, a.dest, a.drop, a.calib)
    print(f"dropped {a.drop} KV groups per layer -> {a.dest}")


if __name__ == "__main__":
    main()

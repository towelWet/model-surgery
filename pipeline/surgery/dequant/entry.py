# entry.py — CLI: dequantize MLX affine checkpoint to bf16 (start point for structural edits).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <quantized_src> <dest>
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.dequant._run import dequant_run

mx.set_default_device(mx.cpu)                               # host work; shared GPU hit watchdog before


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    a = p.parse_args()
    n = dequant_run(a.src, a.dest)
    print(f"dequantized -> {a.dest} ({n} tensors, bf16)")


if __name__ == "__main__":
    main()

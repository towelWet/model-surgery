# entry.py — CLI: weight quantization to MLX affine format by RTN, AWQ or GPTQ.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <bf16_src> <dest> --method rtn|awq|gptq [--bits 4] [--group 64] --calib DIR
#   DIR = profile/moefy/capture.py output (attn / o / mlp taps).
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.wquant._run import wquant_run

mx.set_default_device(mx.cpu)                               # host math; GPU shared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--method", choices=("rtn", "awq", "gptq"), required=True)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--calib", required=True)
    a = p.parse_args()
    wquant_run(a.src, a.dest, a.method, a.bits, a.group, a.calib)
    print(f"{a.method} {a.bits}-bit g{a.group}: {a.dest}")


if __name__ == "__main__":
    main()

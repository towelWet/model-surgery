# entry.py — CLI: convert dense Qwen3 checkpoint into Qwen3MoE with data-fitted routers.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <bf16_src> <quantized_src> <calib_dir> <dest>
#                         [--experts 8] [--topk 2] [--logit-std 0.002]
# calib_dir: pipeline/profile/moefy/capture.py output (runtime probe route, quantized_src served).
# Prints and saves per-layer held-out output error for router / oracle / random expert choice.
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.moefy._moefy import moefy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("quantized")
    p.add_argument("calib")
    p.add_argument("dest")
    p.add_argument("--experts", type=int, default=8)          # E: MLP width splits into E equal experts
    p.add_argument("--topk", type=int, default=2)             # K: experts run per token
    p.add_argument("--logit-std", type=float, default=0.002)  # router logit spread; small = weights ~1/K
    a = p.parse_args()
    moefy(a.src, a.quantized, a.calib, a.dest,
          {"experts": a.experts, "topk": a.topk, "logit_std": a.logit_std})


if __name__ == "__main__":
    main()

# ../moefy/hybrid/entry.py — CLI: dense Qwen3.5/3.6 hybrid -> Qwen3.5-MoE (shared expert + routed experts).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <bf16_src> <quantized_same_model> <calib_dir> <dest> [--experts 8] [--topk 2]
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.moefy.hybrid._hybrid import moefy_hybrid

mx.set_default_device(mx.cpu)                               # host math; GPU shared


def main():
    p = argparse.ArgumentParser()
    for a in ("src", "quantized", "calib", "dest"):
        p.add_argument(a)
    p.add_argument("--experts", type=int, default=8)          # routed experts; +1 width slot = shared
    p.add_argument("--topk", type=int, default=2)             # routed experts run per token
    p.add_argument("--logit-std", type=float, default=0.002)  # router logit spread; small = weights ~1/K
    a = p.parse_args()
    moefy_hybrid(a.src, a.quantized, a.calib, a.dest, {"experts": a.experts, "topk": a.topk, "logit_std": a.logit_std})
    print(f"wrote {a.dest}")


if __name__ == "__main__":
    main()

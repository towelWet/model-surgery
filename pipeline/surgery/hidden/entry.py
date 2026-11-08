# entry.py — CLI: hidden-size pruning (drop least active residual channels everywhere).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <bf16_src> <dest> --keep 768 --calib DIR
#   DIR = profile/moefy/capture.py output ("attn.<i>" and "mlp.<i>" taps).
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.hidden._run import run_hidden

mx.set_default_device(mx.cpu)                               # host work; shared GPU hit watchdog before


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--keep", type=int, required=True)       # new hidden_size
    p.add_argument("--calib", required=True)
    a = p.parse_args()
    run_hidden(a.src, a.dest, a.keep, a.calib)
    print(f"hidden_size -> {a.keep}: {a.dest}")


if __name__ == "__main__":
    main()

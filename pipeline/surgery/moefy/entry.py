# entry.py — CLI for dense FFN → post-hoc MoE partition.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <src_dir> <dest_dir> [--experts 64] [--topk 4]
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # surgery/ parent
sys.path.insert(0, os.path.dirname(__file__))                      # moefy/ siblings
from _moefy import moefy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--experts", type=int, default=64)   # number of expert shards (E)
    p.add_argument("--topk", type=int, default=4)    # experts activated per token (K)
    args = p.parse_args()

    moefy(args.src, args.dest, args.experts, args.topk)


if __name__ == "__main__":
    main()

# entry.py — CLI: passthrough merge (frankenmerge) — stack layer ranges from compatible models.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <model0> [model1 ...] <dest> --slices "0:0-15,1:8-23"
#   Same model twice in --slices duplicates layers (depth up-scaling, e.g. SOLAR-style).
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.passthrough.stack._core import stack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="source models, then output dir")
    p.add_argument("--slices", required=True)              # source:first-last, comma separated, in order
    a = p.parse_args()
    if len(a.paths) < 2:
        p.error("need at least one source and dest")
    *sources, dest = a.paths
    print(f"stacked {len(stack(sources, dest, a.slices))} layers -> {dest}")


if __name__ == "__main__":
    main()

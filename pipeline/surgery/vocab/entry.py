# entry.py — CLI: vocabulary pruning to tokens a corpus (plus BPE closure) needs.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py <src> <dest> --corpus FILE [FILE ...]
#   Corpus should cover every language/domain model must still serve efficiently.
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.vocab._run import run_vocab

mx.set_default_device(mx.cpu)                               # row gather is host work; GPU is shared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--corpus", nargs="+", required=True)
    a = p.parse_args()
    kept, total = run_vocab(a.src, a.dest, a.corpus)
    print(f"vocab {total} -> {kept} tokens: {a.dest}")


if __name__ == "__main__":
    main()

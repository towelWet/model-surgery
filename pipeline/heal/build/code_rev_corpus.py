# code_rev_corpus.py — CLI: materialize heal/data/code_rev_corpus.jsonl
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse
import os
import random
import sys

_HEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pipeline/heal/
for _d in ("code", "rev"):                                           # helper dirs (bare imports)
    sys.path.insert(0, os.path.join(_HEAL, _d))

from _rev_expand import expand_rows
from _write_jsonl import write_jsonl

DEFAULT = os.path.join(_HEAL, "data", "code_rev_corpus.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--out", default=DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rng = random.Random(a.seed)
    rows = expand_rows(rng, a.rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    write_jsonl(a.out, rows)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()

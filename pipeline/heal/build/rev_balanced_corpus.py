# rev_balanced_corpus.py — CLI: 60/40 rev/code corpus (~3000 rows).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse
import os
import random
import sys

_HEAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pipeline/heal/
for _d in ("code", "rev"):                                           # helper dirs (bare imports)
    sys.path.insert(0, os.path.join(_HEAL, _d))

from _balanced_expand import build_balanced
from _write_jsonl import write_jsonl   # was imported from _balanced_expand, which never defined it

DEFAULT_SRC = os.path.join(_HEAL, "data", "code_rev_corpus.jsonl")
DEFAULT_OUT = os.path.join(_HEAL, "data", "rev_balanced_corpus.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--rows", type=int, default=3000)
    p.add_argument("--rev-ratio", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rng = random.Random(a.seed)
    rows = build_balanced(rng, a.src, total=a.rows, rev_ratio=a.rev_ratio)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    write_jsonl(a.out, rows)
    rev_n = sum(1 for r in rows if r.get("tag") in {"disasm", "binary", "review"})
    print(f"wrote {len(rows)} rows (rev-tagged {rev_n}) -> {a.out}")


if __name__ == "__main__":
    main()

# corpus.py — build distillation JSONL (templated tool+chat, offline-first).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage:
#   python3 gen_corpus.py --prompts-only --n 2400
#   python3 gen_corpus.py --n 50
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _corpus_rows import build_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data/distill_corpus.jsonl"))
    p.add_argument("--n", type=int, default=2400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts-only", action="store_true")
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if not args.prompts_only and args.n > 50:
        print("Live gen capped at 50 rows (MEMORY). Use --prompts-only for full corpus.")
        args.n = 50

    n_tool = int(args.n * 0.8)
    chat_fn = None
    if not args.prompts_only:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "profile"))
        from _chat import chat
        chat_fn = chat

    rows = build_rows(rng, n_tool, args.n - n_tool, args.prompts_only, chat_fn)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    mode = "prompts-only" if args.prompts_only else "live"
    print(f"Wrote {len(rows)} rows ({mode}) -> {args.out}")


if __name__ == "__main__":
    main()

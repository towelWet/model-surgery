# convert_stream.py — memory-safe mixed-bit convert (one input shard at a time).
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _convert import run_convert_stream  # noqa: E402
from _recipes import DEFAULT_RECIPE, RECIPES  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Stream mixed-bit convert (bf16-safe)")
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--recipe", default=DEFAULT_RECIPE, choices=list(RECIPES.keys()))
    args = p.parse_args()
    if "Tau" not in args.dest:
        print("WARNING: dest should include Tau")
    run_convert_stream(args.src, args.dest, recipe=args.recipe)


if __name__ == "__main__":
    main()

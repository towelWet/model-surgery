"""Inspect checkpoint metadata and shard bounds without allocating model weights."""
import argparse
from pathlib import Path
from pipeline.checkpoint.summary import summarize
from pipeline.eval.evidence.format import format_scalar


def run(argv):
    parser = argparse.ArgumentParser(prog="ms inspect")
    parser.add_argument("model", help="local checkpoint directory")
    parser.add_argument("--out", help="save a new TOML summary; refuses existing files")
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.model)
        text = "\n".join(f"{key} = {format_scalar(value)}" for key, value in summary.items()) + "\n"
        if args.out:
            with Path(args.out).open("x") as output:
                output.write(text)
        print(text, end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"inspect: {error}")
        return 1


COMMAND = {"name": "inspect", "help": "checkpoint geometry + shard checks, no model load", "run": run}

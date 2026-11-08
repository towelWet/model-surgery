# list.py — `ms list [stage]`: every runnable script per pipeline stage with its header description.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Discovered from disk (no hand-kept table to rot): scripts = .py/.sh without leading underscore.
import os
import pathlib
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.cli._root import pipeline_root

def run(argv):
    parser = argparse.ArgumentParser(prog="ms list")
    parser.add_argument("stages", nargs="*")
    parser.add_argument("--search", default="", help="filter paths and descriptions")
    args = parser.parse_args(argv)
    root = pathlib.Path(pipeline_root())
    stages = {p.name for p in root.iterdir() if p.is_dir() and p.name != "cli" and not p.name.startswith("_")}
    if set(args.stages) - stages:
        parser.error("unknown stage: " + ", ".join(sorted(set(args.stages) - stages)))
    for stage in sorted(stages if not args.stages else args.stages):
        print(f"\n{stage}/")
        for p in sorted((root / stage).rglob("*")):
            if p.suffix not in (".py", ".sh") or p.name.startswith("_") or "__pycache__" in p.parts \
                    or "cases" in p.parts or (p.parent / "__init__.py").exists():
                continue
            lines = p.read_text(errors="ignore").split("\n")
            first = lines[1] if lines[0].startswith("#!") and len(lines) > 1 else lines[0]  # skip shebang
            desc = first.split("—", 1)[-1].strip(' #"') if "—" in first else ""
            if args.search.casefold() not in f"{p.relative_to(root)} {desc}".casefold():
                continue
            print(f"  {str(p.relative_to(root)):<44} {desc[:70]}")
    return 0


COMMAND = {"name": "list", "help": "list runnable scripts per stage with their one-line purpose", "run": run}

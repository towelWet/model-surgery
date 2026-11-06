"""List and validate discovered eval cases without an inference server."""
import argparse
import sys
from pathlib import Path
from pipeline.eval._eval_paths import add_eval_paths
from pipeline.eval._load_cases import load_cases
from pipeline.eval.inspection.cases import check_cases
from pipeline.eval.evidence.fingerprint import fingerprint


def run(argv):
    parser = argparse.ArgumentParser(prog="ms cases")
    parser.add_argument("suites", nargs="*", help="optional suite names")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2] / "eval"
    found = {path.parent.name: path for path in sorted(root.glob("*/run.py")) if (path.parent / "cases").is_dir()}
    unknown = set(args.suites) - set(found)
    if unknown:
        parser.error(f"unknown case suites: {', '.join(sorted(unknown))}")
    failed = 0
    for name, path in found.items():
        if args.suites and name not in args.suites:
            continue
        previous = sys.path[:]
        try:
            add_eval_paths(str(path))
            cases, categories = load_cases(str(path.parent))
            errors = check_cases(cases)
        finally:
            sys.path[:] = previous
        print(f"{name}: {len(cases)} cases, {len(categories)} categories, {len(errors)} errors")
        for error in errors:
            print(f"  {error}")
        failed += len(errors)
    print(f"evaluation_fingerprint = {fingerprint()}")
    return int(bool(failed))


COMMAND = {"name": "cases", "help": "validate + count eval cases offline", "run": run}

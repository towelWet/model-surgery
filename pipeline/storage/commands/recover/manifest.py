# manifest.py — weights recover: read-only inventory of an incomplete store; never deletes.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.storage.inspection import inspect_store


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms weights recover")
    p.add_argument("store")
    a = p.parse_args(argv)
    report = inspect_store(a.store, verify=False)            # read-only; no mutations
    if report.get("complete"):
        print("store complete; nothing to recover")
        return 0
    print(f"incomplete store ({report.get('operation') or 'unknown'}): "
          f"{len(report['missing'])} missing file(s)")
    for rel in report["missing"]:
        print(f"  missing {rel}")
    return 1 if report["missing"] else 0


MANIFEST = {"name": "recover", "help": "read-only inventory of an interrupted store", "run": run}

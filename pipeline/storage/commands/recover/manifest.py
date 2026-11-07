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
    report = inspect_store(a.store, verify=True)             # read-only; verify before claiming recovery
    if report.get("complete"):
        print("store complete; nothing to recover")
        return 0
    print(f"incomplete store ({report.get('operation') or 'unknown'}): "
          f"{len(report['missing'])} missing file(s)")
    for rel in report["missing"]:
        print(f"  missing {rel}")
    for rel in report.get("corrupt", []):                  # hash/size mismatch: was silently unreported
        print(f"  corrupt {rel}")
    for err in report.get("errors", []):                   # manifest validation failures
        print(f"  error {err}")
    return 1                                               # every incomplete or corrupt store is a failure


MANIFEST = {"name": "recover", "help": "read-only inventory of an interrupted store", "run": run}

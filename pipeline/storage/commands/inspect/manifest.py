# manifest.py — weights inspect: report store inventory; --verify hashes every file.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.storage.inspection import inspect_store


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms weights inspect")
    p.add_argument("store")
    p.add_argument("--verify", action="store_true", default=True, help="verify every file (default)")
    p.add_argument("--inventory-only", action="store_true", help="skip hashes; never reports complete")
    a = p.parse_args(argv)
    report = inspect_store(a.store, verify=not a.inventory_only)
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0 if report.get("complete") else 1


MANIFEST = {"name": "inspect", "help": "store inventory; --verify hashes payloads and assets", "run": run}

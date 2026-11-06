# manifest.py — weights split: checkpoint dir -> modular safetensors store.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.storage.split import split_checkpoint


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms weights split")
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    plan = split_checkpoint(a.source, a.destination, dry_run=a.dry_run, resume=a.resume)
    mode = "planned (dry run)" if a.dry_run else "split"
    print(f"{mode}: {len(plan['shards'])} shards, {len(plan['tensors'])} tensors, "
          f"{len(plan['assets'])} assets -> {a.destination}")
    return 0


MANIFEST = {"name": "split", "help": "checkpoint -> modular safetensors store (lossless bytes)", "run": run}

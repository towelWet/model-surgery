# ../moe/split/manifest.py — weights moe-split: MoE checkpoint -> expert-aware store (shared + experts-NNN files).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.storage.experts.split_experts import split_experts


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms weights moe-split")
    p.add_argument("source")                                    # MoE checkpoint dir (config.json names experts)
    p.add_argument("destination")                               # new dir; must not exist
    p.add_argument("--by", default="expert")                    # strategy plugin: storage/experts/strategies/*.py
    p.add_argument("--group", type=int, default=1)              # experts (or layers) per file; 1 = file per unit
    p.add_argument("--dry-run", action="store_true")            # plan + sizes only, no writes
    a = p.parse_args(argv)
    r = split_experts(a.source, a.destination, a.by, a.group, a.dry_run)
    for f, n in r["files"].items():
        print(f"  {f:<28} {n / 2**30:9.3f} GiB")
    mode = "planned (dry run)" if a.dry_run else "split"
    print(f"{mode}: {r['experts']} experts, by {r['strategy']}, group {r['group']}, "
          f"{len(r['files'])} files -> {a.destination}")
    if r["skipped"]:
        print(f"note: subdirs not copied: {', '.join(r['skipped'])}")
    return 0


MANIFEST = {"name": "moe-split", "help": "MoE checkpoint -> expert/layer-aware shards + HF index", "run": run}

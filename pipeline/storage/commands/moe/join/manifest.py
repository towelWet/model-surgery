# ../moe/join/manifest.py — weights moe-join: expert-aware store -> one model.safetensors in original stacked layout.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.storage.experts.join_experts import join_experts


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms weights moe-join")
    p.add_argument("store")                                     # dir holding moe_split.toml
    p.add_argument("destination")                               # new dir; must not exist
    a = p.parse_args(argv)
    r = join_experts(a.store, a.destination)
    print(f"joined: {r['tensors']} tensors, every slice sha256 matched -> {r['dest']}")
    return 0


MANIFEST = {"name": "moe-join", "help": "expert-aware store -> single checkpoint (sha-checked)", "run": run}

# ../prune/experts/manifest.py — gguf prune-experts: keep subset of routed experts, globally or per layer (JSON plan).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.gguf.format.read.gguf import read_gguf
from pipeline.gguf.format.write.gguf import write_gguf
from pipeline.gguf.edit.drop.experts import drop_experts
from pipeline.gguf.edit.kv_entry import kv_entry


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms gguf prune-experts")
    p.add_argument("src")
    p.add_argument("dest", help="new .gguf; must not exist")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--drop", type=int, nargs="+", help="expert indices removed from every layer")
    g.add_argument("--plan", help='JSON {"<layer>": [kept expert indices], ...} covering every MoE layer')
    a = p.parse_args(argv)
    model = read_gguf(a.src)
    arch = kv_entry(model, "general.architecture")["value"]
    n = kv_entry(model, f"{arch}.expert_count")["value"]
    if a.drop:
        kept = [e for e in range(n) if e not in set(a.drop)]
        keep_for = lambda layer: kept                       # same experts gone everywhere
    else:
        plan = {int(k): sorted(v) for k, v in json.loads(Path(a.plan).read_text()).items()}
        keep_for = lambda layer: plan[layer]                # KeyError = plan misses a layer
    count = drop_experts(model, keep_for)
    write_gguf(a.src, a.dest, model)
    print(f"{n} -> {count} experts per layer -> {a.dest}")
    return 0


MANIFEST = {"name": "prune-experts", "help": "drop routed experts (global list or per-layer plan)", "run": run}

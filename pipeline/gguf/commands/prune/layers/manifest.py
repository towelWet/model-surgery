# ../prune/layers/manifest.py — gguf prune-layers: drop whole blocks from a GGUF, byte-copying every survivor.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # project root for pipeline.* imports
from pipeline.gguf.format.read.gguf import read_gguf
from pipeline.gguf.format.write.gguf import write_gguf
from pipeline.gguf.edit.drop.layers import drop_layers


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms gguf prune-layers")
    p.add_argument("src")
    p.add_argument("dest", help="new .gguf; must not exist")
    p.add_argument("--layers", type=int, nargs="+", required=True, help="block indices to remove")
    a = p.parse_args(argv)
    model = read_gguf(a.src)
    keep = drop_layers(model, a.layers)                     # validates before any write
    write_gguf(a.src, a.dest, model)
    print(f"kept {len(keep)} blocks; removed {sorted(set(a.layers))} -> {a.dest}")
    return 0


MANIFEST = {"name": "prune-layers", "help": "remove blocks by index; survivors byte-identical", "run": run}

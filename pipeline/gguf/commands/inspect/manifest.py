# manifest.py — gguf inspect: architecture, key counts, tensor table summary. Reads header only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # project root for pipeline.* imports
from pipeline.gguf.format.read.gguf import read_gguf


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms gguf inspect")
    p.add_argument("file")
    p.add_argument("--tensors", action="store_true", help="list every tensor")
    a = p.parse_args(argv)
    m = read_gguf(a.file)
    arch = next(e["value"] for e in m["kv"] if e["key"] == "general.architecture")
    print(f"version: {m['version']}  arch: {arch}  alignment: {m['alignment']}")
    for e in m["kv"]:
        if e["key"].startswith(arch + ".") and e["type"] != 9:
            print(f"  {e['key']} = {e['value']}")           # scalar hyper-parameters only
    print(f"tensors: {len(m['tensors'])}  payload bytes: {sum(t['nbytes'] for t in m['tensors'])}")
    if a.tensors:
        for t in m["tensors"]:
            print(f"  {t['name']:<40} type={t['type']:<3} dims={t['dims']}")
    return 0


MANIFEST = {"name": "inspect", "help": "header, hyper-parameters and tensor table", "run": run}

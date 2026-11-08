# manifest.py — gguf from-safetensors: HF dir -> GGUF via llama.cpp converter, then llama-quantize.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # project root for pipeline.* imports
from pipeline.gguf.hf.from_safetensors import from_safetensors


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms gguf from-safetensors")
    p.add_argument("hf_dir")
    p.add_argument("out", help="new .gguf")
    p.add_argument("--type", default="Q4_K_M", help="llama-quantize type (F16 = no quantize)")
    a = p.parse_args(argv)
    print(f"wrote {from_safetensors(a.hf_dir, a.out, a.type)}")
    return 0


MANIFEST = {"name": "from-safetensors", "help": "HF safetensors -> GGUF (+ llama-quantize type)", "run": run}

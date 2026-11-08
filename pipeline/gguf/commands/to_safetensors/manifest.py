# manifest.py — gguf to-safetensors: dequantize GGUF to HF bf16 safetensors so any surgery tool applies.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # project root for pipeline.* imports
from pipeline.gguf.hf.to_safetensors import to_safetensors
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer


def run(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ms gguf to-safetensors")
    p.add_argument("src")
    p.add_argument("dest", help="new directory")
    p.add_argument("--hf-ref", required=True, help="original HF dir: config.json + tokenizer files")
    a = p.parse_args(argv)
    import mlx.core as mx
    mx.set_default_device(mx.cpu)                           # dequant + cast is host work
    n = to_safetensors(a.src, a.hf_ref, a.dest)
    copy_tokenizer(a.hf_ref, a.dest)
    print(f"{n} tensors dequantized -> {a.dest}")
    return 0


MANIFEST = {"name": "to-safetensors", "help": "dequantize GGUF to HF bf16 safetensors (llama.cpp gguf-py)", "run": run}

# fuse_adapter.py — fuse LoRA adapters into student checkpoint for serving.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Uses tensor-array-train fuse. mlx_lm fuse was used before and is banned (tensor-array only, ever).
# KNOWN: tensor-array-train had a packed in_features vs hidden mismatch on quantized Tau-56L;
# if fuse fails with that, fix tensor-array-train — never fall back to mlx_lm.
# Usage: python3 fuse_adapter.py --base PATH --adapter DIR --out PATH-Tau-...
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.runtime.command import runtime_command
from pipeline.runtime.environment import runtime_environment


p = argparse.ArgumentParser()
p.add_argument("--base", required=True, help="Student MLX checkpoint")
p.add_argument("--adapter", required=True, help="Adapter directory from tool_distill / general_distill")
p.add_argument("--out", required=True, help="Output checkpoint directory (must not exist)")
args = p.parse_args()

validate_destination(args.out, sources=[args.base, args.adapter])  # fuse CLI creates output
cmd = [
    *runtime_command("train"), "fuse",                                      # flags per `tensor-array-train --help`
    "--model", args.base,
    "--adapter", args.adapter,
    "--out", args.out,
]
print("Starting configured adapter fusion command.")
subprocess.run(cmd, env=runtime_environment(), check=True)
print(f"Fused checkpoint -> {args.out}")

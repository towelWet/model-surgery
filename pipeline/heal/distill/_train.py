# _train.py — tensor-array-train LoRA SFT on student; adds top-32 KL term when teacher cache exists.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.runtime.command import runtime_command
from pipeline.runtime.environment import runtime_environment
from pipeline.runtime.require import require_capability
from pipeline.heal.distill.validate_data import validate_training_data


def train(args, cache):
    evidence = validate_training_data(args.data)
    if evidence["structured_calls"]:
        require_capability("structured_training")
    command = runtime_command("train")                                  # refuse missing trainer before creating output
    os.makedirs(args.adapter_path, exist_ok=True)
    cmd = [
        *command, "train",
        "--model", args.student,
        "--data", args.data,
        "--out", args.adapter_path,
        "--rank", str(args.rank),
        "--alpha", str(args.rank * 2),
        "--steps", str(args.iters),
        "--max-seq", "2048",
        "--lr", str(args.lr),
    ]
    if os.path.exists(cache):
        cmd += ["--teacher-cache", cache]                   # CE + top-32 KL
    else:
        print(f"No teacher cache at {cache}; CE-only SFT. Run cache_teacher_logits.py first.")
    print("Starting configured training command.")
    subprocess.run(cmd, env=runtime_environment(), check=True)
    print(f"\nAdapter -> {args.adapter_path}")

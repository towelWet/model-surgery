# partition.py — profile Empero dense FFN for post-hoc MoE (CPU, no GPU).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Does not load weights into MLX and does not touch :8765.
# Usage: python3 empero_partition.py [model_dir] [--experts 64] [--out path]
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile.empero._profile import empero_profile

DEFAULT = os.environ.get("EMPERO_MODEL_DIR", "")           # HF cache snapshot for Empero 9B
SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(SKILL, "db/qwen/3_5/9B/profile/empero_partition.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", nargs="?", default=DEFAULT)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    result = empero_profile(args.model_dir, args.experts)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    w = result["whittle"]
    print(f"prefix  {result['decoder_prefix']}")
    print(f"shape   H={result['hidden_size']} W={result['intermediate_size']} "
          f"L={result['num_hidden_layers']} tensors={result['n_tensors']}")
    print(f"gate    {result['gate_proj_shape']}")
    print(f"Whittle {w['identity']}  top-4 active={w['active_topk4']}")
    print(f"wrote   {args.out}")


if __name__ == "__main__":
    main()

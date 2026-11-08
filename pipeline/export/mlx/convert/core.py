# convert/core.py — convert HuggingFace / MLX safetensors checkpoint to MLX mixed-bit format.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# First-party convert in export/_convert.py (ported from mlx_lm convert, no subprocess).
# Named recipes from _recipes.py select mixed-bit predicates (boundary layers fatter).
# Usage: python3 mlx_convert.py <src_dir> <dest-Tau-3bit> [--recipe mixed_3_4]
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.export._convert import run_convert  # noqa: E402
from pipeline.export._recipes import DEFAULT_RECIPE, RECIPES  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="Convert MLX/HF checkpoint to mixed-bit MLX safetensors (no mlx_lm)"
    )
    p.add_argument("src", help="Source checkpoint directory")
    p.add_argument("dest", help="Output directory (must not exist; include Tau in name)")
    p.add_argument(
        "--recipe",
        default=DEFAULT_RECIPE,
        choices=list(RECIPES.keys()),
        help="Mixed-bit quant recipe (default: mixed_3_4)",
    )
    args = p.parse_args()

    recipe = RECIPES[args.recipe]
    bits = recipe["default_bits"]
    print(f"Converting {args.src}")
    print(f"  recipe:     {args.recipe}")
    print(
        f"  predicate:  {args.recipe}  ({bits}-bit default, "
        f"{recipe['boundary_bits']}-bit x{recipe['boundary_layers']} boundary layers)"
    )
    print(f"  dest:       {args.dest}")

    if "Tau" not in args.dest:
        print("  WARNING: dest path should include Tau (model-surgery naming rule)")

    run_convert(args.src, args.dest, recipe=args.recipe)


if __name__ == "__main__":
    main()

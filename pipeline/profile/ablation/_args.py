# _args.py — CLI flags for profile/block/ablation.py (modes documented in its header).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse


def ablation_args():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", nargs="?", help="MLX checkpoint directory")
    p.add_argument("--write-spec", action="store_true")
    p.add_argument("--import", dest="do_import", action="store_true")  # `import` is a keyword
    p.add_argument("--from-distance", action="store_true")
    p.add_argument("--score", action="store_true", help="5-prompt local NLL ablation")
    p.add_argument("--score-api", action="store_true", help="same via tensor-array /v1/layer_ablation")
    p.add_argument("--web", action="store_true", help="use WEB_CALIBRATION (HTML/CSS/JS prefixes)")
    return p

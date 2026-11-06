# _args.py — CLI argument parser for neuron_edit.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # surgery/ (checkpoint helpers)
from checkpoint.destination import validate_destination


def parse_args():
    """Parse neuron_edit CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("src")                                    # source checkpoint dir
    p.add_argument("dest")                                   # output checkpoint dir
    p.add_argument("--spec", help="JSON edit spec file")     # full spec overrides individual args
    p.add_argument("--layer", type=int)                      # decoder layer index
    p.add_argument("--proj", default="down_proj.weight")     # projection matrix name
    p.add_argument("--columns", help="Comma-sep col indices")# neuron columns to patch
    p.add_argument("--target", help="Target prompts file")   # behavioral target
    p.add_argument("--baseline", help="Baseline prompts")    # behavioral baseline
    p.add_argument("--alpha", type=float, default=1.0)       # direction scaling factor
    a = p.parse_args()
    validate_destination(a.dest, sources=[a.src])           # Tau name + overlap before any writes
    return a

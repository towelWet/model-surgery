# edit.py — Weight Space Programming: patch neuron columns via activation contrast.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Probes target vs baseline prompts to find behavioral direction,
# then patches specific neuron columns. No gradients, no training.
#
# Reference: rikiyat.github.io/blog/weight-space-programming/
#
# Usage:
#   python3 neuron_edit.py <src> <dest> --layer 29 --columns 512,1024 \
#     --target target.txt --baseline base.txt
#   python3 neuron_edit.py <src> <dest> --spec edit_spec.json
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "profile"))
from _embed import embed                                       # tensor-array embedding endpoint

sys.path.insert(0, os.path.dirname(__file__))
from _args import parse_args                                 # CLI arg parser
from _apply_edit import apply_neuron_edit                          # weight patching engine
from _direction import compute_direction                     # activation contrast

args = parse_args()
os.makedirs(args.dest)                                       # create output dir

with open(os.path.join(args.src, "config.json")) as f:
    cfg = json.load(f)                                       # model config

if args.spec:                                                # full JSON spec mode
    with open(args.spec) as f:
        spec = json.load(f)
    spec["direction"] = mx.array(spec["direction"])          # deserialize direction vector
else:
    spec = compute_direction(args, embed)                    # probe target vs baseline

n = apply_neuron_edit(args.src, args.dest, spec, cfg)        # patch and write checkpoint
print(f"Patched {n} column(s) in layer {spec['layer']} {spec['proj']}")
print(f"Done -> {args.dest}")

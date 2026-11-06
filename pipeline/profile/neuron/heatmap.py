# heatmap.py — CLI: generate HTML heatmap from probe activation data.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Reads probe JSON output, renders interactive HTML neuron heatmap.
#
# Usage:
#   python3 heatmap.py --data activations.json --out map.html
#   python3 heatmap.py --data activations.json --top 30 --out map.html
import argparse
import json
import os
import sys
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))               # profile/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "heatmap"))  # _parse / _render moved to profile/heatmap/
from _parse import parse_layers                      # JSON -> per-layer summaries
from _render import render_heatmap                   # HTML generation

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)                      # probe output JSON
p.add_argument("--out", required=True)                       # output HTML path
p.add_argument("--top", type=int, default=20)                # top-N neurons per layer
p.add_argument("--open", action="store_true")                # open in browser after
args = p.parse_args()

with open(args.data) as f:
    records = json.load(f)
prompts = [r.get("prompt", "") for r in records]

layers = parse_layers(records)
html = render_heatmap(layers, prompts, args.top)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    f.write(html)
print(f"Heatmap: {len(layers)} layers, top-{args.top} -> {args.out}")

if args.open:
    webbrowser.open(f"file://{os.path.abspath(args.out)}")   # open in default browser

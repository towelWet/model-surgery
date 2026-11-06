# recipe.py — CLI: compute steering recipe from target vs baseline prompts.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Probes both prompt sets, computes per-neuron activation contrast,
# outputs TOML recipe with top steering candidates.
#
# Usage:
#   python3 neuron_recipe.py --target target.txt --baseline base.txt --out recipe.toml
#   python3 neuron_recipe.py --target target.txt --baseline base.txt --top 20
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "profile"))            # profile/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "profile", "neuron"))  # _probe_hidden
from _load_lines import load_lines as load_prompts  # shared prompt loader (was _io: stdlib _io always shadowed it)
from _probe_hidden import probe_hidden          # tensor-array probe (was _hidden: profile/_hidden.py shadowed it)

sys.path.insert(0, os.path.dirname(__file__))
from _contrast import compute_contrast                # activation contrast
from _recipe_write import write_recipe                # TOML output

p = argparse.ArgumentParser()
p.add_argument("--target", required=True)                    # target behavior prompts
p.add_argument("--baseline", required=True)                  # baseline prompts
p.add_argument("--top", type=int, default=10)                # top-N steering neurons
p.add_argument("--out", default="recipe.toml")               # output TOML path
args = p.parse_args()

target_p = load_prompts(args.target)
base_p = load_prompts(args.baseline)
print(f"Target: {len(target_p)} prompts, Baseline: {len(base_p)} prompts")

t_acts = [{"prompt": p, "activations": probe_hidden(p)} for p in target_p]
b_acts = [{"prompt": p, "activations": probe_hidden(p)} for p in base_p]

candidates = compute_contrast(t_acts, b_acts, args.top)
n = write_recipe(candidates, args.out)
print(f"Wrote {n} steering candidates -> {args.out}")

for c in candidates[:5]:                                     # preview top 5
    sign = "+" if c["contrast"] > 0 else "-"
    print(f"  L{c['layer']:>2} N{c['neuron']:>5} {sign}{abs(c['contrast']):.4f}")

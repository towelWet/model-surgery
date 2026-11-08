# _rank_blocks.py — turn summed angular distances into ranked cut candidates per block size.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def rank_blocks(model, n_layers, n_prompts, totals):
    result = {"model": model, "n_layers": n_layers, "prompts": n_prompts, "blocks": {}}
    for b, per_start in totals.items():
        ranked = sorted(((v / n_prompts, s) for s, v in per_start.items()))   # ascending = redundant
        result["blocks"][str(b)] = [{"start": s, "end": s + b - 1, "distance": d} for d, s in ranked]
        print(f"\nblock={b:2d}  safest cut -> {ranked[0][1]}..{ranked[0][1]+b-1} d={ranked[0][0]:.4f}"
              f"   | never cut -> {ranked[-1][1]}..{ranked[-1][1]+b-1} d={ranked[-1][0]:.4f}")
    return result

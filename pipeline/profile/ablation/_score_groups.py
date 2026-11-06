# _score_groups.py — rank 4-layer groups by mean NLL delta on calibration prompts.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json

from _ablation_prompts import CALIBRATION             # 5 short prompts (was broken calib._core import)
from _nll import mean_nll
from _ablation_paths import OUT_PATH, n_layers_from_argv
from _mlx_load import load


def score_groups(model_dir, group_size=4, prompts=None):
    """Rank 4-layer groups by mean NLL delta on calibration prompts."""
    prompts = prompts or CALIBRATION
    model, tok = load(model_dir)
    baseline = mean_nll(model, tok, prompts, skip_layers=None)
    n_layers = n_layers_from_argv(model_dir)
    rows = []
    print(f"Local NLL baseline={baseline:.4f} ({len(prompts)} calib prompts)")
    for g in range(n_layers // group_size):
        layers = list(range(g * group_size, (g + 1) * group_size))
        ablated = mean_nll(model, tok, prompts, skip_layers=layers)
        delta = ablated - baseline
        rows.append({
            "group": g,
            "layers": layers,
            "loss_delta": delta,
            "baseline_nll": baseline,
            "ablated_nll": ablated,
        })
        print(f"  group {g} layers={layers} delta={delta:.4f}")
    rows.sort(key=lambda x: x["loss_delta"])                                  # best (least damage) first
    for i, row in enumerate(rows):
        row["rank"] = i
    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote live ranks -> {OUT_PATH}")

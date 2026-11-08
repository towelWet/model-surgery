# _live.py — live NLL scoring via tensor-array /v1/layer_ablation; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Replaces the fake len(text) proxy in block_ablation --score. Uses 5 CALIB prompts only.
import json
import urllib.error

from pipeline.profile.ablation._prompts import CALIBRATION             # 5 short prompts (was broken calib._core import)
from pipeline.profile.ablation._paths import OUT_PATH, n_layers_from_argv
from pipeline.profile.ablation._post import _layer_ablation           # server call (split out)


def score_live(model_dir):
    """Live mean-NLL delta per 4-layer group via /v1/layer_ablation."""
    try:
        base = _layer_ablation(CALIBRATION)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("Server lacks /v1/layer_ablation — rebuild inferserve-model")
        raise
    baseline = base["baseline_nll"]
    n_layers = n_layers_from_argv(model_dir)
    group_size = 4
    rows = []
    print(f"Live NLL baseline={baseline:.4f} ({len(CALIBRATION)} calib prompts)")
    for g in range(n_layers // group_size):
        layers = list(range(g * group_size, (g + 1) * group_size))
        resp = _layer_ablation(CALIBRATION, zero_layers=layers)
        rows.append({
            "group": g,
            "layers": layers,
            "loss_delta": resp["loss_delta"],
            "baseline_nll": resp["baseline_nll"],
            "ablated_nll": resp["ablated_nll"],
        })
        print(f"  group {g} layers={layers} delta={resp['loss_delta']:.4f}")
    rows.sort(key=lambda x: x["loss_delta"])
    for i, row in enumerate(rows):
        row["rank"] = i
    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote live ranks -> {OUT_PATH}")

# _import_results.py — import completed ablation results into ranked JSON.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os
import sys

from pipeline.profile.ablation._paths import OUT_PATH, RESULTS_PATH


def import_results():
    """Parse ablation_results.json, rank by loss_delta, write ablation_ranks.json."""
    if not os.path.exists(RESULTS_PATH):
        print(f"Missing {RESULTS_PATH}")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        raw = json.load(f)
    rows = []
    for r in raw.get("results", raw):
        rows.append({
            "group": r["group"],
            "layers": r["layers"],
            "loss_delta": r["loss_delta"],
            "baseline_nll": r.get("baseline_nll"),
            "ablated_nll": r.get("ablated_nll"),
        })
    rows.sort(key=lambda x: x["loss_delta"])
    for i, row in enumerate(rows):
        row["rank"] = i
    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Imported {len(rows)} groups -> {OUT_PATH}")

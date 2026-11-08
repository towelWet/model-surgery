# _from_distance.py — proxy ablation ranks from block_distance angular data.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os
import sys

from pipeline.profile.ablation._paths import OUT_PATH


def from_distance(model_dir):
    """Build ablation ranks from block_distance.json (proxy when ablation pending)."""
    dist_path = os.path.join(
        os.path.dirname(model_dir) if os.path.basename(model_dir) != "27B" else model_dir,
        "profile", "block_distance.json",
    )
    for cand in (
        dist_path,
        os.path.join(model_dir, "profile", "block_distance.json"),
        os.path.join(os.path.dirname(__file__), "..", "db", "qwen", "3_8", "27B", "profile", "block_distance.json"),
    ):
        if os.path.exists(cand):
            dist_path = cand
            break
    if not os.path.exists(dist_path):
        print(f"No block_distance.json near {model_dir}")
        sys.exit(1)
    with open(dist_path) as f:
        data = json.load(f)
    blocks = data.get("blocks", {})
    rows = []
    entries = blocks.get("4", blocks.get(4, data.get("groups", [])))
    for i, g in enumerate(entries):
        layers = g.get("layers")
        if layers is None and "start" in g:
            layers = list(range(g["start"], g["end"] + 1))
        rows.append({
            "group": g.get("group", i),
            "layers": layers or [],
            "loss_delta": g.get("distance", g.get("sum_distance", 0.0)),
            "source": "block_distance.json",
        })
    rows.sort(key=lambda x: x["loss_delta"])
    for i, row in enumerate(rows):
        row["rank"] = i
    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Proxied angular distance -> {OUT_PATH} ({len(rows)} groups)")

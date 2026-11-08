# _write_spec.py — emit ablation_jobs.json for tensor-array layer gate hook.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json

from pipeline.profile.ablation._prompts import CALIBRATION             # 5 short prompts (was broken calib._core import)
from pipeline.profile.ablation._n_layers import n_layers_from_argv
from pipeline.profile.ablation._paths import SPEC_PATH


def write_spec(model_dir, group_size=4):
    """Write ablation job spec JSON for offline scoring."""
    n_layers = n_layers_from_argv(model_dir)
    n_groups = n_layers // group_size
    jobs = []
    for g in range(n_groups):
        layers = list(range(g * group_size, (g + 1) * group_size))
        jobs.append({
            "job_id": f"group_{g}",
            "group": g,
            "layers": layers,
            "gate": "zero_mlp_and_attn",                 # hook: tensor-array /v1/layer_ablation
            "prompts": CALIBRATION,
            "metric": "mean_nll_delta",
        })
    spec = {
        "model": model_dir,
        "group_size": group_size,
        "n_groups": n_groups,
        "format_version": 1,
        "jobs": jobs,
        "result_schema": {
            "job_id": "str",
            "baseline_nll": "float",
            "ablated_nll": "float",
            "loss_delta": "float",
        },
    }
    with open(SPEC_PATH, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote {len(jobs)} ablation jobs -> {SPEC_PATH}")

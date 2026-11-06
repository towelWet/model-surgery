# _ablation_post.py — one POST to tensor-array /v1/layer_ablation (baseline, or with zeroed layers).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _api import _post


def _layer_ablation(prompts, zero_layers=None):
    body = {"prompts": prompts}
    if zero_layers:
        body["zero_layers"] = zero_layers           # omitted = baseline NLL only
    return _post("layer_ablation", body)

# _depth_score.py — per-layer NLL scoring for MoD routing decisions.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# For each layer, measures NLL with vs without that layer.
# Uses tensor-array /v1/layer_ablation endpoint.
from _api import _post


def score_layers(prompts, threshold):
    """Score each layer's importance across prompt corpus.

    Returns dict with skippable/essential layer lists and per-layer scores.
    """
    resp = _post("models", {})                             # get model info for layer count
    info = resp.get("data", [{}])[0]
    n_layers = info.get("num_hidden_layers", 64)

    scores = {}                                            # layer_idx -> mean NLL delta
    for layer in range(n_layers):
        deltas = []
        for prompt in prompts:
            body = {
                "model": "current",
                "prompt": prompt,
                "ablate_layer": layer,                     # tensor-array ablation endpoint
            }
            try:
                r = _post("layer_ablation", body)
                deltas.append(r.get("nll_delta", 0.0))    # NLL(ablated) - NLL(intact)
            except Exception:
                deltas.append(float("inf"))                # layer is critical if probe fails
        mean_d = sum(deltas) / len(deltas) if deltas else float("inf")
        scores[layer] = mean_d
        tag = "SKIP" if mean_d < threshold else "KEEP"
        print(f"  layer {layer:3d}: NLL delta {mean_d:+.4f} [{tag}]")

    skippable = sorted(k for k, v in scores.items() if v < threshold)
    essential = sorted(k for k, v in scores.items() if v >= threshold)
    return {
        "n_layers": n_layers,
        "threshold": threshold,
        "n_prompts": len(prompts),
        "skippable": skippable,
        "essential": essential,
        "scores": {str(k): v for k, v in scores.items()},
    }

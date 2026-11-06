# _parse.py — parse probe JSON into per-layer activation summaries.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def parse_layers(records):
    """Aggregate activation records into per-layer mean absolute values.

    Returns list of dicts: [{"mean_abs": [float, ...], "layer": int}, ...]
    """
    if not records:
        return []
    layers = {}                                              # layer_idx -> running sums
    count = len(records)
    for rec in records:
        acts = rec.get("activations", {})
        for layer_key, neurons in acts.items():
            li = int(layer_key) if str(layer_key).isdigit() else hash(layer_key) % 999
            if li not in layers:
                layers[li] = [0.0] * len(neurons)
            for ni, val in enumerate(neurons):
                if ni < len(layers[li]):
                    layers[li][ni] += abs(val) / count       # running mean of |activation|

    result = []
    for li in sorted(layers):
        result.append({"layer": li, "mean_abs": layers[li]})
    return result

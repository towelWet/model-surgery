# _parse.py — parse probe JSON into per-layer activation summaries.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def parse_layers(records, absolute=True):
    """Aggregate activation records into per-layer signed or absolute means.

    Returns list of dicts: [{"mean_abs": [float, ...], "layer": int}, ...]
    """
    if not records:
        return []
    layers = {}                                              # layer_idx -> running sums
    counts = {}
    for rec in records:
        acts = rec.get("activations", {})
        for layer_key, neurons in acts.items():
            if not str(layer_key).isdigit():
                raise ValueError(f"activation layer ID must be numeric: {layer_key}")
            li = int(layer_key)
            if li not in layers:
                layers[li] = [0.0] * len(neurons)
                counts[li] = 0
            if len(neurons) != len(layers[li]):
                raise ValueError(f"inconsistent activation width in layer {li}")
            counts[li] += 1
            for ni, val in enumerate(neurons):
                if ni < len(layers[li]):
                    layers[li][ni] += abs(val) if absolute else val

    result = []
    for li in sorted(layers):
        field = "mean_abs" if absolute else "mean"
        result.append({"layer": li, field: [v / counts[li] for v in layers[li]]})
    return result

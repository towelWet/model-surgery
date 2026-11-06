# _contrast.py — compute activation contrast between target and baseline prompts.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# For each layer, computes mean(target) - mean(baseline) per neuron.
# Returns top-N neurons ranked by absolute contrast magnitude.


def compute_contrast(target_records, baseline_records, top_n=10):
    """Compute per-layer neuron contrast, return ranked steering candidates.

    Returns: [{layer, neuron, contrast, target_mean, baseline_mean}, ...]
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "profile", "heatmap"))  # _parse lives in profile/heatmap/
    from _parse import parse_layers                  # reuse layer parser

    t_layers = parse_layers(target_records)
    b_layers = parse_layers(baseline_records)

    candidates = []
    for tl in t_layers:
        li = tl["layer"]
        bl = next((b for b in b_layers if b["layer"] == li), None)
        if not bl:
            continue
        for ni in range(min(len(tl["mean_abs"]), len(bl["mean_abs"]))):
            diff = tl["mean_abs"][ni] - bl["mean_abs"][ni]   # signed contrast
            candidates.append({
                "layer": li, "neuron": ni,
                "contrast": diff,
                "target_mean": tl["mean_abs"][ni],
                "baseline_mean": bl["mean_abs"][ni],
            })

    candidates.sort(key=lambda c: abs(c["contrast"]), reverse=True)
    return candidates[:top_n]

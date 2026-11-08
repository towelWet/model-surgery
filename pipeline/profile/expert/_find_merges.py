# _find_merges.py — expert pairs that fire together often enough to merge (Jaccard co-activation).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def _find_merges(lh, coact_layer, n_experts, threshold=0.7):
    """Find expert pairs with Jaccard co-activation > threshold."""
    merges = []
    for ei in range(n_experts):
        for ej in range(ei + 1, n_experts):                    # upper triangle: each pair once
            co = coact_layer[ei, ej]
            if co == 0:
                continue
            union = lh[ei] + lh[ej] - co                       # |A ∪ B| = |A| + |B| - |A ∩ B|
            j = co / max(union, 1)
            if j > threshold:
                merges.append({"pair": [int(ei), int(ej)],
                               "jaccard": round(float(j), 3)})
    return sorted(merges, key=lambda x: -x["jaccard"])[:10]    # top 10 pairs per layer

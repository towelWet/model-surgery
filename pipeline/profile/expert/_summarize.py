# _summarize.py — build JSON summary from routing capture arrays.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np

from _find_merges import _find_merges                  # co-activation merge pairs (split out)


def summarize_routing(model_path, n_layers, n_experts, top_k,
                      total_tok, n_prompts, hits, coact):
    """Summarize hits[n_layers,n_experts] + coact into JSON-serializable dict."""
    result = {"model": model_path, "n_layers": n_layers, "n_experts": n_experts,
              "top_k": top_k, "total_tokens": total_tok, "prompts": n_prompts,
              "per_layer": {}}
    for li in range(n_layers):
        lh = hits[li]; tot = lh.sum()
        if tot == 0:
            continue                                           # non-MoE layer
        freq = lh / tot
        ranked = sorted(range(n_experts), key=lambda e: freq[e])  # least-used first
        merges = _find_merges(lh, coact[li], n_experts)
        result["per_layer"][str(li)] = {
            "never_used": int((lh == 0).sum()),
            "top10pct_share": round(float(np.sort(freq)[-max(1, n_experts//10):].sum()), 4),
            "prune_order": ranked,                             # ALL experts, least-used first
            "merge_candidates": merges,
        }
    return result


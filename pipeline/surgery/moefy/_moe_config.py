# _moe_config.py — write dest config.json describing the new MoE layout (shared + E experts, top-K).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def write_moe_config(dest, cfg, tc, E, K, S, H):
    tc["num_experts"] = E
    tc["num_experts_per_tok"] = K
    tc["moe_intermediate_size"] = S                         # width of each routed expert
    tc["shared_expert_intermediate_size"] = H               # shared expert keeps H neurons
    tc["norm_topk_prob"] = True                             # renormalize top-K router weights
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)                         # tc is a view into cfg (VLM nests it)

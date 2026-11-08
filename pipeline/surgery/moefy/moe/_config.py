# moe/_config.py — config.json for converted checkpoint: dense Qwen3 -> Qwen3MoE.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def write_moe_config(dest, dense_cfg, experts, topk):
    """Copy dense config; switch architecture and add routed-expert fields runtime loader reads."""
    cfg = dict(dense_cfg)
    cfg["architectures"] = ["Qwen3MoeForCausalLM"]
    cfg["model_type"] = "qwen3_moe"
    cfg["num_experts"] = experts
    cfg["num_experts_per_tok"] = topk
    cfg["moe_intermediate_size"] = dense_cfg["intermediate_size"] // experts
    cfg["norm_topk_prob"] = True                              # runtime always renormalises top-K
    cfg["decoder_sparse_step"] = 1                            # every layer is MoE
    cfg["mlp_only_layers"] = []
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

# ../moe/combine/_config.py — Qwen3MoE config for combined experts: each expert keeps full MLP width.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def write_combine_config(dest, base_cfg, experts, topk):
    cfg = dict(base_cfg)
    cfg["architectures"] = ["Qwen3MoeForCausalLM"]
    cfg["model_type"] = "qwen3_moe"
    cfg["num_experts"] = experts
    cfg["num_experts_per_tok"] = topk
    cfg["moe_intermediate_size"] = base_cfg["intermediate_size"]  # whole fine-tune MLP per expert
    cfg["norm_topk_prob"] = True                              # runtime renormalises top-K weights
    cfg["decoder_sparse_step"] = 1
    cfg["mlp_only_layers"] = []
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

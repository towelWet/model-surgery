# _mixtral_config.py — config.json for converted checkpoint: dense Llama/Mistral -> Mixtral.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# HF Mixtral fields: num_local_experts, num_experts_per_tok, intermediate_size = per-expert width.
# Runtime registers MixtralForCausalLM on Llama pack (llama::ffn routes block_sparse_moe layers).
import json
import os


def write_mixtral_config(dest, dense_cfg, experts, topk):
    cfg = dict(dense_cfg)
    cfg["architectures"] = ["MixtralForCausalLM"]
    cfg["model_type"] = "mixtral"
    cfg["num_local_experts"] = experts
    cfg["num_experts_per_tok"] = topk
    cfg["intermediate_size"] = dense_cfg["intermediate_size"] // experts  # Mixtral: width of one expert
    cfg["router_aux_loss_coef"] = 0.0                       # no training-time balancing loss here
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

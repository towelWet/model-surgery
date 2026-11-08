# ../moefy/hybrid/_config.py — config.json for dense Qwen3.5 -> Qwen3.5-MoE (shared expert + routed experts).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Field names copied from real Qwen3.6-35B-A3B config (text_config holds the decoder dims).
import copy
import json
import os


def write_hybrid_config(dest, dense_cfg, experts, topk, width):
    cfg = copy.deepcopy(dense_cfg)
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]  # runtime ArchRegistrar key
    cfg["model_type"] = "qwen3_5_moe"
    t = cfg.setdefault("text_config", {})
    t.update(model_type="qwen3_5_moe_text", num_experts=experts, num_experts_per_tok=topk,
             moe_intermediate_size=width, shared_expert_intermediate_size=width,
             norm_topk_prob=True)                              # runtime renormalises top-K anyway
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

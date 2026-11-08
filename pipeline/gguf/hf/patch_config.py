# patch_config.py — HF config.json updated to shapes actually present in an (edited) GGUF.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# GGUF edits (layer/expert/width pruning) change block count and widths; reference HF config
# still holds original values. Read them back from GGUF metadata so converter round-trips.


def patch_config(cfg, meta, arch, vocab_rows):
    keys = {"block_count": "num_hidden_layers", "embedding_length": "hidden_size",
            "feed_forward_length": "intermediate_size", "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads", "expert_count": "num_experts",
            "expert_used_count": "num_experts_per_tok", "expert_feed_forward_length": "moe_intermediate_size"}
    out = dict(cfg)
    for g, h in keys.items():
        if f"{arch}.{g}" in meta:
            out[h] = int(meta[f"{arch}.{g}"])
    out["vocab_size"] = vocab_rows
    out.pop("quantization", None)                             # dequantized output is plain bf16
    out.pop("quantization_config", None)
    return out

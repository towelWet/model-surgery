# name.py — HF tensor name for one GGUF tensor name, via gguf-py's own TensorNameMap.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# TensorNameMap lists every alias (gpt-neox, falcon, llama, "attention.wq" ...) for each GGUF
# name. Alias present in HF reference's own tensor names wins; without reference weights,
# HF Llama/Qwen naming (model.layers.N.self_attn|mlp|*_layernorm, embed_tokens, norm, lm_head).
import re

LLAMA_STYLE = re.compile(r"^(lm_head|model\.(embed_tokens|norm|layers\.\d+\.(self_attn|mlp|input_layernorm|post_attention_layernorm)))")


def hf_name(tmap, gguf_name, known=frozenset()):
    base, suffix = gguf_name.rsplit(".", 1)                   # "blk.0.ffn_down", "weight"
    aliases = [k for k, v in tmap.mapping.items() if v[1] == base and k != base]
    hit = [k for k in aliases if f"{k}.{suffix}" in known] or [k for k in aliases if LLAMA_STYLE.match(k)]
    if not hit:
        raise KeyError(f"no HF name for GGUF tensor {gguf_name}")
    return f"{min(hit, key=len)}.{suffix}"                    # shortest = plain name, not nested variant

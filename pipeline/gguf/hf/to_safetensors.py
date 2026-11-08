# to_safetensors.py — dequantize GGUF into HF-layout bf16 safetensors (config/tokenizer from HF reference).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Uses llama.cpp's gguf-py for reading + dequantizing (every ggml quant type it supports), so
# no k-quant code of our own. Whole model is held in memory as bf16 (one file). Llama arch:
# converter permuted q/k rows for rope layout; unpermute_qk restores HF order.
import json
import os
import mlx.core as mx
from .llamacpp_paths import llamacpp_paths
from .name import hf_name
from .meta import gguf_meta
from .patch_config import patch_config
from .ref_keys import ref_keys
from .unpermute_qk import unpermute_qk

# Tensors llama.cpp's converter derives from config (no HF tensor): rope_freqs = Llama 3
# rope_scaling factors. HF rebuilds them from config.json, so they are dropped, not renamed.
DERIVED = {"rope_freqs.weight"}


def to_safetensors(src, hf_ref, dest):
    llamacpp_paths()                                          # puts gguf-py on sys.path
    import gguf
    if os.path.exists(dest):
        raise FileExistsError(dest)
    r = gguf.GGUFReader(src)
    meta = gguf_meta(r)
    arch = meta["general.architecture"]
    heads = {"attn_q.weight": meta.get(f"{arch}.attention.head_count"),           # q permuted with n_head
             "attn_k.weight": meta.get(f"{arch}.attention.head_count_kv")} if arch == "llama" else {}
    enum = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}[arch]
    tmap = gguf.TensorNameMap(enum, int(meta[f"{arch}.block_count"]))
    known, out, vocab = ref_keys(hf_ref), {}, 0
    for t in (t for t in r.tensors if t.name not in DERIVED):
        w = mx.array(gguf.quants.dequantize(t.data, t.tensor_type)).astype(mx.bfloat16)
        h = heads.get(t.name.split(".", 2)[-1])                 # "blk.N.attn_q.weight" -> "attn_q.weight"
        w = unpermute_qk(w, int(h)) if h else w
        out[hf_name(tmap, t.name, known)] = w
        vocab = w.shape[0] if t.name == "token_embd.weight" else vocab
    os.makedirs(dest)
    mx.save_safetensors(os.path.join(dest, "model.safetensors"), out, metadata={"format": "pt"})
    cfg = patch_config(json.load(open(os.path.join(hf_ref, "config.json"))), meta, arch, vocab)
    cfg["tie_word_embeddings"] = "lm_head.weight" not in out
    json.dump(cfg, open(os.path.join(dest, "config.json"), "w"), indent=2)
    return len(out)

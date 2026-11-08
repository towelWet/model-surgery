# _run.py — bf16 checkpoint -> MLX affine-quantized checkpoint by RTN, AWQ or GPTQ.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Decoder linears at `bits`; embed_tokens / lm_head at 8-bit RTN for every method (per-tensor
# config override), so comparisons isolate linear-quantization quality. Norms stay bf16.
import glob, json, os
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import copy_tokenizer
from pipeline.surgery.wquant.layer._core import wquant_layer

HEAD = ("model.embed_tokens", "lm_head")                       # 8-bit RTN for all methods


def wquant_run(src, dest, method, bits, group, calib):
    dest = validate_destination(dest, sources=[src, calib])
    cfg = json.load(open(os.path.join(src, "config.json")))
    if "quantization" in cfg:
        raise ValueError("source already quantized: start from bf16")
    files = sorted(glob.glob(os.path.join(calib, "*.safetensors")))
    lazy = {k: v for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))) for k, v in mx.load(f).items()}
    out = {k: v for k, v in lazy.items() if ".layers." not in k and not k.startswith(HEAD)}
    for h in HEAD:
        if h + ".weight" in lazy:
            q, s, b = mx.quantize(lazy[h + ".weight"], group_size=group, bits=8)
            out.update({h + ".weight": q, h + ".scales": s, h + ".biases": b})
    for li in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{li}."
        out.update({k: v for k, v in lazy.items() if k.startswith(p) and ("_norm." in k)})  # q_norm/k_norm
        out.update(wquant_layer(lazy, li, files, method, bits, group))
        mx.eval(list(out.values()))
        print(f"layer {li} {method}", flush=True)
    os.makedirs(dest)
    mx.save_safetensors(str(dest / "model.safetensors"), out, metadata={"format": "mlx"})
    cfg["quantization"] = {"group_size": group, "bits": bits, **{h: {"group_size": group, "bits": 8} for h in HEAD}}
    json.dump(cfg, open(dest / "config.json", "w"), indent=2)
    copy_tokenizer(src, dest)

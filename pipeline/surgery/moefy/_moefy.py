# _moefy.py — dense Qwen3 / Llama -> Qwen3MoE / Mixtral: experts from co-activation clusters, routers fit on data.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Inputs: bf16 dense source (exact MLP weights), the same model quantized by runtime's own
# quantizer (every non-MLP tensor is copied from it, so dense-vs-MoE comparisons isolate the
# MLP change), and capture.py calibration files. Last quarter of files is held out.
import glob
import json
import os
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer
from pipeline.surgery.moefy.expert._counts import check_counts
from pipeline.surgery.moefy._convert_layer import convert_layer
from pipeline.surgery.moefy._non_mlp import non_mlp
from pipeline.surgery.moefy.moe._target import moe_target


def moefy(src, quant_base, calib, dest, plan):
    dest = validate_destination(dest, sources=[src, quant_base, calib])
    cfg = json.load(open(os.path.join(quant_base, "config.json")))
    check_counts(plan["experts"], plan["topk"])
    q = cfg["quantization"]
    if cfg["intermediate_size"] % (plan["experts"] * q["group_size"]):  # down_proj groups run along expert width
        raise ValueError(f"expert width {cfg['intermediate_size']}/{plan['experts']} must be a multiple of {q['group_size']}")
    block, write_config = moe_target(cfg)                    # Qwen3 -> Qwen3-MoE, Llama/Mistral -> Mixtral
    plan = {**plan, "group": q["group_size"], "bits": q["bits"], "block": block}  # match base quantization
    files = sorted(glob.glob(os.path.join(calib, "*.safetensors")))
    if len(files) < 4:
        raise ValueError("need >= 4 calibration files (3/4 fit, 1/4 held out)")
    cut = len(files) * 3 // 4
    src_w = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        src_w.update(mx.load(f))                              # lazy: MLP tensors load on use
    out = non_mlp(quant_base)                                 # attention, norms, embed (CPU, materialised)
    report = []
    for li in range(cfg["num_hidden_layers"]):
        tensors, row = convert_layer(src_w, li, files[:cut], files[cut:], plan)
        out.update(tensors)
        mx.eval(list(tensors.values()))                       # bound graph per layer: one-shot eval at save timed out GPU on Llama 3B
        report.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(dest)
    mx.save_safetensors(str(dest / "model.safetensors"), out, metadata={"format": "mlx"})
    copy_tokenizer(quant_base, dest)
    write_config(dest, cfg, plan["experts"], plan["topk"])
    json.dump({"plan": plan, "layers": report}, open(dest / "moefy_report.json", "w"), indent=2)
    return report

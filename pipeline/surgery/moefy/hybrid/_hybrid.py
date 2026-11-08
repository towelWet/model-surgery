# ../moefy/hybrid/_hybrid.py — dense Qwen3.5 (hybrid linear/full attention) -> Qwen3.5-MoE with shared expert.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Inputs as moefy: bf16 dense (exact MLP weights), same model quantized (every non-MLP tensor copied
# from it, so comparison isolates MLP change), capture.py calibration (last quarter held out).
# Width split: intermediate = (experts + 1) x expert width; one width-slot is shared expert.
import glob, json, os
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer
from pipeline.surgery._layer_prefix import decoder_layer_prefix
from pipeline.surgery.moefy.hybrid._convert_hybrid_layer import convert_hybrid_layer
from pipeline.surgery.moefy.hybrid._config import write_hybrid_config


def moefy_hybrid(src, quant_base, calib, dest, plan):
    dest = validate_destination(dest, sources=[src, quant_base, calib])
    cfg = json.load(open(os.path.join(quant_base, "config.json")))
    t, q = cfg.get("text_config", cfg), cfg["quantization"]
    width, rem = divmod(t["intermediate_size"], plan["experts"] + 1)
    if rem or width % q["group_size"]:
        raise ValueError(f"intermediate {t['intermediate_size']} must split into {plan['experts']}+1 multiples of {q['group_size']}")
    plan = {**plan, "shared": width, "group": q["group_size"], "bits": q["bits"]}
    files = sorted(glob.glob(os.path.join(calib, "*.safetensors")))
    if len(files) < 4:
        raise ValueError("need >= 4 calibration files (3/4 fit, 1/4 held out)")
    cut = len(files) * 3 // 4
    src_w = {k: v for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))) for k, v in mx.load(f).items()}
    out = {k: v for f in sorted(glob.glob(os.path.join(quant_base, "*.safetensors")))
           for k, v in mx.load(f).items() if ".mlp." not in k}  # attention, norms, embed, vision
    pre, report = decoder_layer_prefix(list(src_w), t["num_hidden_layers"]), []
    for li in range(t["num_hidden_layers"]):
        tensors, row = convert_hybrid_layer(src_w, pre, li, files[:cut], files[cut:], plan)
        out.update(tensors); report.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(dest)
    mx.save_safetensors(str(dest / "model.safetensors"), out, metadata={"format": "mlx"})
    copy_tokenizer(quant_base, dest)
    write_hybrid_config(dest, cfg, plan["experts"], plan["topk"], width)
    json.dump({"plan": plan, "layers": report}, open(dest / "moefy_report.json", "w"), indent=2)
    return report

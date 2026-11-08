# ../moe/combine/_run.py — dense fine-tunes of one base -> Qwen3MoE with one expert per fine-tune.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Non-MLP tensors (attention, norms, embeddings) come from `base` (quantized): fine-tunes'
# attention deltas are dropped unless caller passes e.g. a linear merge as base.
import glob, json, os
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer
from pipeline.surgery.moe.combine._router_rows import router_rows
from pipeline.surgery.moe.combine._layer import combine_layer
from pipeline.surgery.moe.combine._config import write_combine_config


def combine(base, experts, calib_dirs, dest, topk, logit_std=2.0):
    if len(experts) != len(calib_dirs) or len(experts) < 2 or not 0 < topk <= len(experts):
        raise ValueError("need >= 2 experts, one calibration dir each, 0 < topk <= experts")
    dest = validate_destination(dest, sources=[base, *experts, *calib_dirs])
    cfg = json.load(open(os.path.join(base, "config.json")))
    group, bits = cfg["quantization"]["group_size"], cfg["quantization"]["bits"]
    lazy = [{k: v for f in sorted(glob.glob(os.path.join(e, "*.safetensors"))) for k, v in mx.load(f).items()}
            for e in experts]
    sets = [sorted(glob.glob(os.path.join(c, "*.safetensors"))) for c in calib_dirs]
    out = {}
    for f in sorted(glob.glob(os.path.join(base, "*.safetensors"))):
        out.update({k: v for k, v in mx.load(f).items() if ".mlp." not in k})
    for li in range(cfg["num_hidden_layers"]):
        out.update(combine_layer(li, lazy, router_rows(sets, li, logit_std), group, bits))
        mx.eval(list(out.values()))                          # bound graph per layer
    os.makedirs(dest)
    mx.save_safetensors(str(dest / "model.safetensors"), out, metadata={"format": "mlx"})
    copy_tokenizer(base, dest)
    write_combine_config(dest, cfg, len(experts), topk)

# _run.py — hidden-size (embedding-channel) pruning of a bf16 checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Keeps `keep` most important residual channels everywhere they appear. RMSNorm then averages
# over fewer channels, so activations shift: expect quality loss until healed (Minitron heals
# with distillation). Unknown tensor touching hidden size is refused rather than guessed.
import glob, json, os, shutil
import numpy as np
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import copy_tokenizer, load_index
from pipeline.surgery.hidden._channel_importance import channel_importance
from pipeline.surgery.hidden._axis import hidden_axis


def run_hidden(src, dest, keep, calib, group=64):
    dest = validate_destination(dest, sources=[src, calib])
    cfg = json.load(open(os.path.join(src, "config.json")))
    h = cfg["hidden_size"]
    if not 0 < keep < h or keep % group:
        raise ValueError(f"keep must be < {h} and a multiple of {group} (quantization groups)")
    wmap = load_index(src)["weight_map"]
    if any(k.endswith(".scales") for k in wmap):
        raise ValueError("quantized checkpoint: prune bf16 source first, then quantize")
    files = sorted(glob.glob(os.path.join(calib, "*.safetensors")))
    score = channel_importance(files, cfg["num_hidden_layers"])
    idx = mx.array(np.sort(np.argsort(-score)[:keep]))       # original channel order preserved
    os.makedirs(dest)
    for shard in sorted(set(wmap.values())):
        w, out = mx.load(os.path.join(src, shard)), {}
        for k, v in w.items():
            ax = hidden_axis(k, v.shape, h)
            if ax is None and h in v.shape and v.ndim > 1:
                raise ValueError(f"{k} {v.shape} has hidden-size dim but no known role")
            out[k] = v if ax is None else mx.take(v, idx, axis=ax)
            mx.eval(out[k])
        mx.save_safetensors(str(dest / shard), out, metadata={"format": "mlx"})
    cfg["hidden_size"] = keep
    json.dump(cfg, open(dest / "config.json", "w"), indent=2)
    for f in ("model.safetensors.index.json", "generation_config.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dest / f)
    copy_tokenizer(src, dest)
    return score

# _run.py — MLX affine-quantized checkpoint -> bf16 checkpoint (same shard names).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Structural edits refuse quantized sources ("prune bf16, then quantize"); this gives them a
# bf16 starting point when only quantized weights exist on disk. Rounding error of original
# quantization stays baked in: score edits against this dequantized dense, never fp original.
import glob, json, os, shutil
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import copy_tokenizer, write_index
from pipeline.surgery.dequant._tensors import dequant_tensors


def dequant_run(src, dest):
    dest = validate_destination(dest, sources=[src])
    cfg = json.load(open(os.path.join(src, "config.json")))
    if "quantization" not in cfg:
        raise ValueError("source is not MLX-quantized (no config.quantization)")
    os.makedirs(dest)
    wmap = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        out = dequant_tensors(mx.load(f), cfg)                 # lazy load; one shard in memory
        mx.save_safetensors(str(dest / os.path.basename(f)), out, metadata={"format": "mlx"})
        wmap.update({k: os.path.basename(f) for k in out})
    for k in ("quantization", "quantization_config"):         # bf16 now: runtime must not unpack
        cfg.pop(k, None)
        cfg.get("text_config", {}).pop(k, None)
    json.dump(cfg, open(dest / "config.json", "w"), indent=2)
    write_index(str(dest), {}, wmap)
    for f in ("generation_config.json", "chat_template.jinja", "preprocessor_config.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dest / f)
    copy_tokenizer(src, dest)
    return len(wmap)

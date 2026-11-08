# _run.py — low-rank (SVD / ASVD) compression of decoder linears in a bf16 checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Default: dense A@B reconstruction (any runtime can score quality) plus lowrank/factors.safetensors
# (`<key>.lr_a` [out, r], `<key>.lr_b` [r, in]); subdirectory so loaders globbing *.safetensors
# skip it. --factored: factors replace weights in shards (runtime lowrank_linear), real size cut.
import glob, json, os, shutil
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import copy_tokenizer, load_index, write_index
from pipeline.surgery.lowrank._shard import lowrank_shard


def run_lowrank(src, dest, ratio, roles, calib, alpha, factored=False):
    dest = validate_destination(dest, sources=[src] + ([calib] if calib else []))
    idx = load_index(src)
    if any(k.endswith(".scales") for k in idx["weight_map"]):
        raise ValueError("quantized checkpoint: factor bf16 source first, then quantize")
    files = sorted(glob.glob(os.path.join(calib, "*.safetensors"))) if calib else []
    lazy = {k: v for s in sorted(set(idx["weight_map"].values())) for k, v in mx.load(os.path.join(src, s)).items()}
    ctx = {"ratio": ratio, "roles": roles, "files": files, "alpha": alpha, "factored": factored, "norms": {}}
    os.makedirs(dest)
    side, report, wmap = {}, [], {}
    for shard in sorted(set(idx["weight_map"].values())):
        out, s, rows = lowrank_shard([k for k, v in idx["weight_map"].items() if v == shard], lazy, ctx)
        mx.save_safetensors(str(dest / shard), out, metadata={"format": "mlx"})
        side.update(s); report += rows; wmap.update({k: shard for k in out})
    if side:
        os.makedirs(dest / "lowrank")
        mx.save_safetensors(str(dest / "lowrank" / "factors.safetensors"), side)
    write_index(str(dest), idx.get("metadata", {}), wmap)        # factored keys differ from source
    for f in ("config.json", "generation_config.json", "chat_template.jinja"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dest / f)
    copy_tokenizer(src, dest)
    json.dump({"ratio": ratio, "alpha": alpha, "asvd": bool(files), "factored": factored, "tensors": report},
              open(dest / "lowrank_report.json", "w"), indent=1)
    return report

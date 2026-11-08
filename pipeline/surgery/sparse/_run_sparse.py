# _run_sparse.py — unstructured pruning of a bf16 checkpoint, shard by shard, with per-tensor report.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Methods: magnitude (|W|), wanda (|W| x captured input-channel norm), sparsegpt (OBS mask +
# error compensation from captured-input Hessians). Default: wanda with calib, else magnitude.
# Zeros do not shrink safetensors or speed MLX (dense kernels); gains need compression or
# sparse kernels elsewhere. Quantize AFTER pruning: affine quant zero is not exact zero.
import glob, json, os, shutil
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import copy_tokenizer, load_index
from pipeline.surgery.sparse._target import sparse_target
from pipeline.surgery.sparse.prune._key import prune_key

ROLES = ("q", "k", "v", "o", "gate", "up", "down")       # all decoder linears


def run_sparse(src, dest, sparsity, nm, calib, roles=ROLES, method=None):
    method = method or ("wanda" if calib else "magnitude")  # old behaviour when unspecified
    if method != "magnitude" and not calib:
        raise ValueError(f"{method} needs --calib")
    dest = validate_destination(dest, sources=[src] + ([calib] if calib else []))
    idx = load_index(src)
    if any(k.endswith(".scales") for k in idx["weight_map"]):
        raise ValueError("quantized checkpoint: prune bf16 source first, then quantize")
    files = sorted(glob.glob(os.path.join(calib, "*.safetensors"))) if calib else []
    lazy = {}
    for shard in sorted(set(idx["weight_map"].values())):
        lazy.update(mx.load(os.path.join(src, shard)))      # lazy arrays: load on first use
    os.makedirs(dest)
    ctx, report = {"method": method, "sparsity": sparsity, "nm": nm, "files": files, "cache": {}}, []
    for shard in sorted(set(idx["weight_map"].values())):
        out = {}
        for key in [k for k, s in idx["weight_map"].items() if s == shard]:
            tgt = sparse_target(key)
            tgt = tgt if tgt and tgt[1] in roles else None          # role filter (e.g. MLP only)
            out[key], zero = prune_key(key, lazy[key], tgt, lazy, ctx) if tgt else (lazy[key], 0.0)
            if tgt:
                report.append({"key": key, "method": method, "zero": zero})
        mx.save_safetensors(str(dest / shard), out, metadata={"format": "mlx"})
    for f in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dest / f)
    copy_tokenizer(src, dest)
    json.dump({"method": method, "sparsity": sparsity, "nm": nm, "roles": sorted(roles), "tensors": report}, open(dest / "sparse_report.json", "w"), indent=1)
    return report

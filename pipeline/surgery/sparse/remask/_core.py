# remask/_core.py — copy healed weights, re-zeroing every position zero in pruned checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# LoRA fuse adds a dense low-rank delta, refilling pruned zeros. Re-applying mask after each
# heal keeps target sparsity: prune -> heal -> fuse -> remask, repeatable at rising sparsity
# (iterative pruning). Not masked-gradient sparse training: zeros move nothing during heal.
import json, os, shutil
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import copy_tokenizer, load_index
from pipeline.surgery.sparse._target import sparse_target


def remask(pruned, healed, dest):
    dest = validate_destination(dest, sources=[pruned, healed])
    pmap, hmap = load_index(pruned)["weight_map"], load_index(healed)["weight_map"]
    if set(pmap) != set(hmap):
        raise ValueError("pruned and healed checkpoints have different tensor keys")
    plazy = {}
    for shard in sorted(set(pmap.values())):
        plazy.update(mx.load(os.path.join(pruned, shard)))
    os.makedirs(dest)
    zeros = 0
    for shard in sorted(set(hmap.values())):
        w = mx.load(os.path.join(healed, shard))
        for k in w:
            if sparse_target(k):
                keep = plazy[k] != 0
                w[k] = mx.where(keep, w[k], mx.zeros_like(w[k]))
                mx.eval(w[k])                                 # per tensor: bounded GPU work
                zeros += int((~keep).sum().item())
        mx.save_safetensors(str(dest / shard), w, metadata={"format": "mlx"})
    for f in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        if os.path.exists(os.path.join(healed, f)):
            shutil.copy2(os.path.join(healed, f), dest / f)
    copy_tokenizer(healed, dest)
    return zeros

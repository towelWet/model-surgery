# _save_pruned.py — write pruned checkpoint: same shard layout, edited tensors swapped in.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json, os, shutil
import mlx.core as mx
from pipeline.surgery.shard_io import copy_tokenizer, write_index


def save_pruned(src, dest, wmap, lazy, new, cfg):
    """new: edited key -> array (others copied from lazy); cfg already updated by caller."""
    os.makedirs(dest)
    for shard in sorted(set(wmap.values())):
        out = {k: new.get(k, lazy[k]) for k, s in wmap.items() if s == shard}
        mx.save_safetensors(str(dest / shard), out, metadata={"format": "mlx"})
    json.dump(cfg, open(dest / "config.json", "w"), indent=2)
    write_index(str(dest), {}, wmap)
    for f in ("generation_config.json", "chat_template.jinja"):   # runtime reads template from file
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dest / f)
    copy_tokenizer(src, dest)

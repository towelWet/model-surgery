# drop/layers.py — remove whole blocks from a parsed GGUF: tensors, renumbering, per-layer metadata.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import re

from ..kv_entry import kv_entry

BLK = re.compile(r"^blk\.(\d+)\.(.+)$")                   # llama.cpp per-layer tensor naming


def drop_layers(model, removed):
    arch = kv_entry(model, "general.architecture")["value"]
    count = kv_entry(model, f"{arch}.block_count")
    old = count["value"]
    removed = set(removed)
    if not removed or not removed <= set(range(old)):
        raise ValueError(f"layers must be a non-empty subset of 0..{old - 1}")
    nextn = kv_entry(model, f"{arch}.nextn_predict_layers")
    if nextn and nextn["value"] and max(removed) >= old - nextn["value"]:
        raise ValueError("refusing to cut MTP/nextn layers at end of stack")
    keep = [i for i in range(old) if i not in removed]
    renumber = {old_i: new_i for new_i, old_i in enumerate(keep)}  # survivors close ranks in order
    tensors = []
    for t in model["tensors"]:
        m = BLK.match(t["name"])
        if m and int(m.group(1)) in removed:
            continue
        if m:
            t["name"] = f"blk.{renumber[int(m.group(1))]}.{m.group(2)}"
        tensors.append(t)
    model["tensors"] = tensors
    for e in model["kv"]:                                   # per-layer arrays (e.g. head counts) shrink too
        if e["key"].startswith(arch + ".") and e["type"] == 9 and len(e["value"]["items"]) == old:
            e["value"]["items"] = [e["value"]["items"][i] for i in keep]
    count["value"] = len(keep)
    return keep

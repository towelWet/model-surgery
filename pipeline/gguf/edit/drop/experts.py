# drop/experts.py — keep a subset of routed experts per layer by slicing expert-dim byte ranges.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import re

from ..kv_entry import kv_entry

BLK = re.compile(r"^blk\.(\d+)\.")
ROUTED = ("_exps.", "ffn_gate_inp.", "exp_probs_b.")      # tensors whose last dim is n_expert; shexp untouched


def drop_experts(model, keep_for_layer):
    """keep_for_layer(layer) -> sorted expert indices to keep; same count for every layer (GGUF rule)."""
    arch = kv_entry(model, "general.architecture")["value"]
    total = kv_entry(model, f"{arch}.expert_count")
    used = kv_entry(model, f"{arch}.expert_used_count")
    if total is None:
        raise ValueError("not an MoE GGUF: no expert_count")
    n, sizes = total["value"], set()
    for t in model["tensors"]:
        m = BLK.match(t["name"])
        if not m or not any(r in t["name"] for r in ROUTED):
            continue
        if t["dims"][-1] != n:
            raise ValueError(f"{t['name']}: last dim {t['dims'][-1]} != expert_count {n}")
        keep = keep_for_layer(int(m.group(1)))
        if sorted(set(keep)) != list(keep) or not keep or keep[-1] >= n:
            raise ValueError(f"layer {m.group(1)}: keep list must be sorted unique indices < {n}")
        sizes.add(len(keep))
        (pos, nbytes), = t["chunks"]                        # untouched tensor: one contiguous range
        step = nbytes // n                                  # bytes per expert slice (outermost dim)
        chunks = []
        for e in keep:                                      # merge adjacent kept experts into one range
            if chunks and chunks[-1][0] + chunks[-1][1] == pos + e * step:
                chunks[-1] = (chunks[-1][0], chunks[-1][1] + step)
            else:
                chunks.append((pos + e * step, step))
        t["chunks"], t["dims"] = chunks, t["dims"][:-1] + [len(keep)]
        t["nbytes"] = step * len(keep)                      # keep size field truthful for inspect
    if len(sizes) != 1:
        raise ValueError(f"every layer must keep same expert count; got {sorted(sizes)}")
    kept = sizes.pop()
    if used and kept < used["value"]:
        raise ValueError(f"keeping {kept} experts < {used['value']} used per token")
    total["value"] = kept
    return kept

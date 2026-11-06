# harvest.py — teacher top-K over every assistant position of every row; writes NPZ + meta.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys

from .assistant_span import assistant_span
from .position_topk import position_topk
from .position_weight import position_weight
from .write_cache import write_cache


def harvest(rows, tok, out_npz, out_meta):
    idx_all, prob_all, lab_all, w_all, offsets = [], [], [], [], [0]
    meta_rows = []
    drop = {"no_assistant": 0, "align": 0}                 # align kept for log shape; padding makes it always 0
    for ri, row in enumerate(rows):
        messages = row.get("messages", row)
        tools = row.get("tools")
        full_ids, start, end = assistant_span(tok, messages, tools)
        if full_ids is None:
            drop["no_assistant"] += 1
            continue
        n_pos = end - start
        for pi in range(n_pos):                            # one server call per assistant token
            ids_k, probs_k = position_topk(tok, messages, tools, full_ids, start, start + pi)
            idx_all.append(ids_k); prob_all.append(probs_k)
            lab_all.append(full_ids[start + pi])           # ground-truth label at this position
            w_all.append(position_weight(pi, n_pos))
        offsets.append(len(lab_all))                       # row boundary into flat arrays
        meta_rows.append({"row": len(meta_rows), "n_pos": n_pos, "messages": messages})
        if ri % 10 == 0:
            print(f"  row {ri}/{len(rows)}  kept={len(meta_rows)}", flush=True)
    if not lab_all:
        print("No rows cached.", drop)
        sys.exit(1)
    write_cache(out_npz, out_meta, offsets, idx_all, prob_all, lab_all, w_all, meta_rows)
    print(f"Wrote {out_npz}  rows={len(meta_rows)}  positions={len(lab_all)}  drop={drop}")

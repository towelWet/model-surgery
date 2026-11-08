# write_cache.py — save harvested arrays to NPZ + per-row meta JSONL (format in _constants.py).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json

import numpy as np


def write_cache(out_npz, out_meta, offsets, idx_all, prob_all, lab_all, w_all, meta_rows):
    np.savez_compressed(
        out_npz,
        offsets=np.array(offsets, dtype=np.int32),
        idx=np.array(idx_all, dtype=np.int32),
        prob=np.array(prob_all, dtype=np.float32),
        label=np.array(lab_all, dtype=np.int32),
        weight=np.array(w_all, dtype=np.float32),
    )
    with open(out_meta, "w") as f:
        for m in meta_rows:
            f.write(json.dumps(m) + "\n")                  # meta.jsonl: line i = NPZ row i

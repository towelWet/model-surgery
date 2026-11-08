# ../merge/_ties/run_merge.py — full merge: verify configs, merge tensors, write config + shards + index.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os

import numpy as np

from ._shared import load_index
from .tensors import merge_tensors
from .config import merged_configuration
from pipeline.surgery.checkpoint.destination import validate_destination
from .preflight import merge_preflight
from pipeline.surgery.checkpoint.write.groups import write_groups

LABELS = {"dare": "dare_task_arithmetic"}                 # historical label readers already match


def run_merge(sources, dest, spec, seed, bf16_only):
    """sources[0] is base; spec = {method, density, lambda, weights}."""
    validate_destination(dest, sources=sources)
    maps = merge_preflight(sources, spec)                   # also refuses fine-tune-only keys
    rng = np.random.default_rng(seed)                       # only DARE variants draw from it
    index = load_index(sources[0])
    only_base = set(maps[0]) - set.intersection(*(set(m) for m in maps[1:]))
    if only_base:
        print(f"  {len(only_base)} base-only keys (e.g. vision) copied unchanged")
    groups = merge_tensors(sources, maps, only_base, rng, spec, bf16_only)
    meta = dict(index.get("metadata", {}))
    meta["merge_method"] = LABELS.get(spec["method"], spec["method"])
    meta["merge_density"] = str(spec["density"])
    meta["merge_base"] = os.path.basename(sources[0])
    meta["merge_others"] = ",".join(os.path.basename(s) for s in sources[1:])
    cfg = merged_configuration(sources[0], bf16_only)
    write_groups(sources[0], dest, groups, {**index, "metadata": meta}, cfg)
    print(f"\nDone -> {dest}")
    print(f"  method: {spec['method']}  density={spec['density']}  lambda={spec['lambda']}  weights={spec['weights']}")

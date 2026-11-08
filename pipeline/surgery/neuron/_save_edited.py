# _save_edited.py — write edited checkpoint to dest.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os

import sys as _sys2, os as _os2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.shard_io import copy_tokenizer
from pipeline.surgery._pack_shards import save_weight_dict


def save_edited(src, dest, cfg, out, wmap, idx):
    """Copy tokenizer, write config, save weight dict."""
    copy_tokenizer(src, dest)
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_weight_dict(dest, out, wmap, idx.get("metadata", {}))

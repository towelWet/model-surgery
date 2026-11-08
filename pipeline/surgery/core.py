# core.py — backwards-compatible re-export of _keys and shard_io (no importers left as of 2026-11-06).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# New code should import from _keys.py and shard_io.py directly.
# This file exists only so older callers continue to work.

import os, sys                              # root bootstrap needs both
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery._keys import remap_key, shard_name                      # key remapping + shard naming
from pipeline.surgery.shard_io import load_index, write_index, copy_tokenizer, load_shard_cache  # file I/O

SRC = None                                                       # set via caller or env; never hardcode

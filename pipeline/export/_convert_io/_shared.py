# _shared.py — module-level settings, path setup and shared imports for this package (was _module.py).
import glob
import json
import os
import shutil
import sys
import mlx.core as mx
# Package lives in export/_convert_io/ — surgery/ is two levels up.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.shard_io import copy_tokenizer, load_index, load_shard_cache, write_index
MAX_FILE_SIZE_GB = 5
EVAL_EVERY = 30

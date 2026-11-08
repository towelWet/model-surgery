# shard_io.py — re-export safetensors I/O from shard/ split files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os, sys                              # root bootstrap needs both
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.shard.load._index import load_index                 # index parser
from pipeline.surgery.shard._write_index import write_index               # index writer
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer         # tokenizer copier
from pipeline.surgery.shard.load._cache import load_shard_cache            # cached shard loader

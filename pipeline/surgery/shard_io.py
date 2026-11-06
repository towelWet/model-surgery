# shard_io.py — re-export safetensors I/O from shard/ split files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from shard._load_index import load_index                 # index parser
from shard._write_index import write_index               # index writer
from shard._copy_tokenizer import copy_tokenizer         # tokenizer copier
from shard._load_cache import load_shard_cache            # cached shard loader

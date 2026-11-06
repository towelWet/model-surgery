# _st_raw.py — re-export safetensors raw readers (header/index/bytes/hash); no logic.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Reading byte ranges keeps RAM flat even for 5 GB shards on a 32 GB machine.
from _st_header import header                  # one shard header
from _st_index import index                    # tensor name -> shard + range
from _st_tensor_bytes import tensor_bytes      # raw bytes of one tensor
from _file_sha256 import file_sha256           # whole-file hash

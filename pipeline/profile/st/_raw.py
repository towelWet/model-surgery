# st/_raw.py — re-export safetensors raw readers (header/index/bytes/hash); no logic.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Reading byte ranges keeps RAM flat even for 5 GB shards on a 32 GB machine.
from pipeline.profile.st._header import header                  # one shard header
from pipeline.profile.st._index import index                    # tensor name -> shard + range
from pipeline.profile.st._tensor_bytes import tensor_bytes      # raw bytes of one tensor
from pipeline.profile._file_sha256 import file_sha256           # whole-file hash

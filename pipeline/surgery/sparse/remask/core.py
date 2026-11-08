# remask/core.py — CLI: restore pruned zeros after heal/fuse (sparsity-preserving heal).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 remask/core.py <pruned_dir> <healed_fused_dir> <dest>
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
import mlx.core as mx
from pipeline.surgery.sparse.remask._core import remask

mx.set_default_device(mx.cpu)                               # host work; shared GPU hit watchdog before

if len(sys.argv) != 4:
    sys.exit("usage: remask/core.py <pruned_dir> <healed_fused_dir> <dest>")
print(f"re-zeroed {remask(*sys.argv[1:4])} weights -> {sys.argv[3]}")

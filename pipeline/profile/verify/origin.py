# origin.py — do two INDEPENDENT quantizations dequantize to the same weights?
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Historically sampled four fixed Qwen tensors. Now discovers common quantized matrices.
# Cosine is similarity evidence, not proof of checkpoint provenance or common origin.
# Reads only sampled rows, never whole shards; reports a shuffled-row control too.
# Usage: python3 origin.py <model_a> <model_b> [--tensor PREFIX] [--rows N]
import os, sys
import argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # verify/ siblings
from _deq_rows import deq                                   # sampled-row dequantizer
from _row_cosine import cos                                 # per-row cosine
from pipeline.checkpoint.tensors import tensor_index
from pipeline.profile.verify.similarity.selection import select_tensors

parser = argparse.ArgumentParser(description="Sample weight similarity; not provenance proof")
parser.add_argument("model_a")
parser.add_argument("model_b")
parser.add_argument("--tensor", action="append", help="tensor prefix; repeat to select")
parser.add_argument("--rows", type=int, default=256)
parser.add_argument("--row-start", type=int, default=0)
parser.add_argument("--limit", type=int, default=8, help="maximum sampled matrices; 0 = all")
args = parser.parse_args()
if args.rows < 2 or args.row_start < 0:
    parser.error("--rows must be >=2; --row-start must be nonnegative")
left, right = tensor_index(args.model_a), tensor_index(args.model_b)
names = select_tensors(left, right, args.tensor, args.limit)
for name in names:
    count = min(args.rows, left[name + ".weight"][1]["shape"][0] - args.row_start,
                right[name + ".weight"][1]["shape"][0] - args.row_start)
    if count < 2:
        parser.error(f"{name}: insufficient rows for shuffled control")
    a, b = deq(args.model_a, name, args.row_start, count), deq(args.model_b, name, args.row_start, count)
    same, ctrl = cos(a, b), cos(a, np.roll(b, 1, axis=0))               # control: row i vs row i-1
    print(f"{name.split('model.')[-1]:32s} matched cos mean {same.mean():.4f} min {same.min():.4f}"
          f" | shuffled control {ctrl.mean():+.4f}")

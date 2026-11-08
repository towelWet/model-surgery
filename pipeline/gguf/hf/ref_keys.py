# ref_keys.py — tensor names of HF reference checkpoint (index or safetensors headers; no tensor reads).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import glob
import json
import os
import struct


def ref_keys(hf_ref):
    idx = os.path.join(hf_ref, "model.safetensors.index.json")
    if os.path.exists(idx):
        return set(json.load(open(idx))["weight_map"])
    keys = set()
    for f in glob.glob(os.path.join(hf_ref, "*.safetensors")):
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]             # header length, little-endian u64
            keys |= set(json.loads(h.read(n))) - {"__metadata__"}
    return keys

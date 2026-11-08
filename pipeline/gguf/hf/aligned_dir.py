# aligned_dir.py — checkpoint dir the llama.cpp converter can read: itself, or an aligned staging copy.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Staging copy is kept beside output (like f16 intermediate); source dir is never modified.
import glob
import os
import shutil
from .is_aligned import is_aligned
from .realign_file import realign_file


def aligned_dir(hf_dir, staging):
    shards = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    if all(is_aligned(s) for s in shards):
        return hf_dir                                         # HF uploads are already aligned
    os.makedirs(staging)                                      # fails if exists: no silent reuse
    for name in os.listdir(hf_dir):
        s, d = os.path.join(hf_dir, name), os.path.join(staging, name)
        if name.endswith(".safetensors"):
            realign_file(s, d)
        elif os.path.isfile(s) and not name.startswith("."):  # config, tokenizer, index; skip AppleDouble
            shutil.copy2(s, d)
    return staging

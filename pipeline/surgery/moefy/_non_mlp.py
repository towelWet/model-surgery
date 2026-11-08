# _non_mlp.py — every non-MLP tensor (attention, norms, embed) of quantized base, read and materialised on CPU.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Left lazy, these were first evaluated inside save_safetensors in one GPU command buffer;
# Llama 3B top-4 hit Metal watchdog there (top-8 passed on same code: at limit, not fixed by retry).
import glob
import os
import mlx.core as mx


def non_mlp(quant_base):
    out = {}
    with mx.stream(mx.cpu):                                   # host copy: keeps save off shared GPU
        for f in sorted(glob.glob(os.path.join(quant_base, "*.safetensors"))):
            out.update({k: v for k, v in mx.load(f).items() if ".mlp." not in k})
        mx.eval(list(out.values()))                           # read now, so save only writes bytes
    return out

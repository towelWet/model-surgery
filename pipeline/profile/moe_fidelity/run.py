# run.py — CLI: measure how well each partition reproduces dense MLP output.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Was profile/moe_fidelity.py beside this package (same name clash as moe_partition).
# Usage: python3 profile/moe_fidelity/run.py <model_dir> <stats_dir> [--layers 0,14,27,41,55]
import json, os, sys
import numpy as np, mlx.core as mx
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery._spec import parse_spec
from pipeline.profile.moe_fidelity._down_proj import down_proj                             # dequantized weight loader
from pipeline.profile.moe_fidelity._rel import rel                                         # relative error metric

model, stats = sys.argv[1], sys.argv[2]
layers = sorted(parse_spec(sys.argv[sys.argv.index("--layers") + 1])) \
    if "--layers" in sys.argv else [0, 14, 27, 41, 55]
KS, BITS, GS = (4, 8, 16), 3, 64                            # top-k sweep + quant params
wm = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]

print(f"{'layer':>5} {'mode':>6} " + " ".join(f"k={k:<3}" for k in KS) + "  requant_err")
for i in layers:
    D = down_proj(model, wm, i, GS, BITS)
    a = mx.array(np.load(os.path.join(stats, f"L{i:02d}", "sample.npy")).astype(np.float32))
    for mode in ("group", "neuron"):
        part = np.load(os.path.join(stats, "partition", mode, f"L{i:02d}.npz"))
        experts = mx.array(part["experts"])                                  # [64, 192]
        mass = mx.abs(a)[:, experts].sum(-1)                                 # [N, 64] expert mass
        errs = []
        for k in KS:
            top = mx.argpartition(-mass, k - 1, axis=1)[:, :k]              # oracle top-k experts
            keep = mx.zeros(a.shape)
            idx = experts[top].reshape(a.shape[0], -1)                       # [N, k*192] neuron ids
            keep = mx.put_along_axis(keep, idx, mx.ones(idx.shape), axis=1)
            keep[:, mx.array(part["shared"])] = 1.0                          # shared expert always on
            errs.append(rel(D, a, keep))
        rq = ""
        if mode == "neuron":                                                 # permute + requantize cost
            perm = mx.array(np.concatenate([part["shared"], part["experts"].ravel()]))
            Dp = D[:, perm]
            Dq = mx.dequantize(*mx.quantize(Dp, group_size=GS, bits=BITS), group_size=GS, bits=BITS)
            inv = mx.argsort(perm)
            rq = f"{rel(Dq[:, inv], a, mx.ones(a.shape), D_ref=D):.4f}"
        print(f"{i:>5} {mode:>6} " + " ".join(f"{e:.4f}" for e in errs) + f"  {rq}", flush=True)
    mx.clear_cache()

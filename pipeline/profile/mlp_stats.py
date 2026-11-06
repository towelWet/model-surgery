# mlp_stats.py — one pass over calibration text, collecting everything MoE conversion needs.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Per layer, saved as float32 .npy under <out>/L<nn>/:
#   xtx.npy       [5120, 5120]  sum of m_in outer products     -> ridge router fit
#   xtg.npy       [5120, 272]   m_in x |act| summed per 64-neuron quant group -> router targets
#   neuron.npy    [17408]       sum of |act| per neuron        -> shared-expert selection
#   sample.npy    [N, 17408]    every 8th token's activations  -> clustering + coverage study
# Groups of 64 match down_proj quantization groups, so group-level experts stay byte-exact.
# Usage: python3 mlp_stats.py <model_dir> <out_dir> [--check]
import json, os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # profile/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer"))  # _tap lives in layer/
from _mlx_load import load
from calib._corpus import chunks
from _tap import capture_mlp
from _mlp_accumulator import MlpAccumulator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "surgery", "mlp"))  # _hw lives in surgery/mlp/
from _hw import mlp_hw                   # Tau 5120/17408 or Empero 4096/12288

model_dir, out = sys.argv[1], sys.argv[2]
check = "--check" in sys.argv
model, tok = load(model_dir)
data = chunks(tok)
n_layers = len(getattr(model, "language_model", model).layers)
cfg = json.load(open(os.path.join(model_dir, "config.json")))
H, W = mlp_hw(cfg.get("text_config", cfg))   # do not hardcode Tau 27B dims
G = 64
stats = MlpAccumulator(n_layers, H, W, G)
tokens = 0

t0 = time.time()
for k, ids in enumerate(data):
    capture_mlp(model, mx.array(ids).reshape(1, -1), stats.on_layer, check=(check and k == 0))
    tokens += len(ids)
    mx.clear_cache()
    print(f"  chunk {k + 1}/{len(data)}  tokens {tokens}  {tokens / (time.time() - t0):.0f} tok/s", flush=True)

stats.save(out)
json.dump({"model": os.path.basename(model_dir.rstrip("/")), "tokens": tokens, "chunks": len(data),
           "layers": n_layers, "group": G}, open(os.path.join(out, "meta.json"), "w"), indent=2)
print(f"done: {tokens} tokens, {n_layers} layers -> {out}")

# capture.py — save every layer's linear-layer inputs for calibration text, via runtime probe route.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Dense-to-MoE needs what each MLP actually sees: post-attention-norm hidden states.
# The runtime writes them server-side (POST /v1/internals/linear_inputs, probe env gate on)
# as <out>/<k>.safetensors holding float32 "mlp.<i>", "attn.<i>", "o.<i>". No local model load.
#
# Usage: python3 capture.py <out_dir> [--pieces 64] [--max-tokens 256] [--texts rows.jsonl]
#   --texts: chat JSONL rows (message contents joined) instead of built-in pieces, e.g. router
#   prompts per expert domain for moe_combine.
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile._api import _post
from pipeline.profile.moefy._pieces import pieces

p = argparse.ArgumentParser()
p.add_argument("out")                                      # empty or new directory
p.add_argument("--pieces", type=int, default=64)           # 64 x 256 tokens ~ 16k calibration tokens
p.add_argument("--max-tokens", type=int, default=256)      # per-piece truncation; bounds file size
p.add_argument("--texts", default=None)                    # chat JSONL; overrides --pieces
a = p.parse_args()

out = os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)
if os.listdir(out):
    sys.exit(f"{out} is not empty; refusing to mix calibration runs")
total = 0
texts = (["\n".join(m["content"] for m in json.loads(l)["messages"]) for l in open(a.texts) if l.strip()]
         if a.texts else pieces(a.pieces))
for k, text in enumerate(texts):
    reply = _post("internals/linear_inputs", {"input": text, "out": os.path.join(out, f"{k:04d}.safetensors"),
                                           "max_tokens": a.max_tokens})
    total += reply["tokens"]
    print(f"piece {k}: {reply['tokens']} tokens x {reply['layers']} layers")
print(f"captured {total} tokens into {out}")

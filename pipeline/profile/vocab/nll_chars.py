# nll_chars.py — held-out loss per CHARACTER, comparable across different tokenizers.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Per-token NLL is not comparable after vocabulary pruning: same text splits into more,
# easier-to-predict tokens. Total nats over identical text, divided by its characters, is.
# Each piece is posted alone (route averages per prompt) with no truncation; token count
# comes from served model's own tokenizer.json. First token has no prediction (S-1 targets).
#
# Usage: python3 nll_chars.py MODEL_DIR [--skip 64] [--count 32]
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from tokenizers import Tokenizer
from pipeline.profile._api import _post
from pipeline.profile.moefy._pieces import pieces

p = argparse.ArgumentParser()
p.add_argument("model_dir")                                # served checkpoint (for its tokenizer)
p.add_argument("--skip", type=int, default=64)             # pieces used by capture / vocab corpus
p.add_argument("--count", type=int, default=32)
a = p.parse_args()

tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
nats = chars = tokens = 0
for text in pieces(a.skip + a.count)[a.skip:]:
    n = len(tok.encode(text, add_special_tokens=False).ids)
    reply = _post("layer_ablation", {"prompts": [text], "max_tokens": 4 * n})  # no truncation
    nats += reply["baseline_nll"] * (n - 1)
    chars, tokens = chars + len(text), tokens + n
print(f"heldout chars={chars} tokens={tokens} nats_per_char={nats / chars:.4f} nll_per_token={nats / (tokens - a.count):.4f}")

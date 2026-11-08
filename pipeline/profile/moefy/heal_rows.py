# heal_rows.py — write calibration pieces as chat JSONL rows for dense->MoE distillation.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Trainer masks loss to assistant turns, so each piece is an assistant reply to a fixed
# "continue" prompt. Pieces 0..count-1 = same text capture.py fitted routers on; nll.py
# scores later pieces, so heal never trains on evaluation text.
#
# Usage: python3 heal_rows.py <out.jsonl> [--count 64]
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile.moefy._pieces import pieces

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--count", type=int, default=64)
a = p.parse_args()

if os.path.exists(a.out):
    sys.exit(f"{a.out} exists; refusing to overwrite")
with open(a.out, "w") as f:
    for text in pieces(a.count):
        row = {"messages": [{"role": "user", "content": "Continue this text."},
                            {"role": "assistant", "content": text}]}
        f.write(json.dumps(row) + "\n")
print(f"wrote {a.count} rows to {a.out}")

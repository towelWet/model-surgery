# lineage.py — prove a pruned checkpoint is its parent minus removed layers, nothing more.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Check A: parent shards hash to the lfs_sha256 Hugging Face recorded at download time.
# Check B: every child tensor is byte-identical to the parent tensor it was copied from,
#          after undoing the layer renumbering. Any fine-tune, swap, or edit breaks this.
# Usage: python3 verify_lineage.py <parent_dir> <child_dir> "<removed spec>"
import glob, hashlib, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "surgery"))
from _st_raw import index, tensor_bytes, file_sha256
from _spec import parse_spec

parent, child, spec = sys.argv[1], sys.argv[2], sys.argv[3]
removed = parse_spec(spec)

print("== A: parent download integrity ==")
trees = glob.glob(os.path.join(parent, ".cache/huggingface/trees/*.json"))
files = json.load(open(trees[0]))["files"] if trees else {}
for shard in sorted(k for k in files if k.endswith(".safetensors")):
    got = file_sha256(os.path.join(parent, shard))
    ok = got == files[shard]["lfs_sha256"]
    print(f"  {shard}: {'MATCH' if ok else 'MISMATCH'}  {got[:16]}…")

print("\n== B: child tensors vs parent ==")
kept = [i for i in range(10_000) if i not in removed]           # child idx -> parent idx
pidx, cidx = index(parent), index(child)
same = diff = missing = 0
for name, loc in cidx.items():
    m = re.match(r"^(language_model\.model\.layers\.)(\d+)(\..+)$", name)
    pname = f"{m.group(1)}{kept[int(m.group(2))]}{m.group(3)}" if m else name
    if pname not in pidx:
        missing += 1; print("  MISSING in parent:", name); continue
    pe, ce = pidx[pname][1], loc[1]
    eq = pe["dtype"] == ce["dtype"] and pe["shape"] == ce["shape"] and \
         hashlib.sha256(tensor_bytes(pidx[pname])).digest() == hashlib.sha256(tensor_bytes(loc)).digest()
    same += eq; diff += not eq
    if not eq: print("  DIFFERS:", name, "<-", pname)
print(f"  child tensors: {len(cidx)} | identical: {same} | differ: {diff} | missing: {missing}")
print(f"  parent tensors not carried over: {len(pidx) - len(cidx)} (expected = removed layers only)")

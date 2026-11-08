"""Serialize a split plan into manifest.toml text (schema_version 1)."""
from .escape import toml_str


def manifest_text(plan):
    out = [f"schema_version = {plan['schema_version']}",
           f"kind = {toml_str(plan['kind'])}",
           f"guarantees = {toml_str(plan['guarantees'])}",
           f"source_fingerprint = {toml_str(plan['source_fingerprint'])}"]
    for key in ("shards", "tensors", "assets"):              # empty arrays must sit in root table —
        if not plan[key]:                                    # after any [[table]] a bare key binds wrong
            out.append(f"{key} = []")
    out.append("")
    for s in plan["shards"]:
        out += ["[[shards]]", f"path = {toml_str(s['path'])}", f"header = {toml_str(s['header'])}",
                f"sha256 = {toml_str(s['sha256'])}", f"size = {s['size']}", ""]
    for t in plan["tensors"]:
        shape = "[" + ", ".join(str(d) for d in t["shape"]) + "]"
        out += ["[[tensors]]", f"name = {toml_str(t['name'])}", f"dtype = {toml_str(t['dtype'])}",
                f"shape = {shape}", f"shard = {toml_str(t['shard'])}",
                f"offsets = [{t['offsets'][0]}, {t['offsets'][1]}]", f"file = {toml_str(t['file'])}",
                f"sha256 = {toml_str(t['sha256'])}", f"nbytes = {t['nbytes']}", ""]
    for a in plan["assets"]:
        out += ["[[assets]]", f"path = {toml_str(a['path'])}", f"file = {toml_str(a['file'])}",
                f"sha256 = {toml_str(a['sha256'])}", f"size = {a['size']}", ""]
    return "\n".join(out)

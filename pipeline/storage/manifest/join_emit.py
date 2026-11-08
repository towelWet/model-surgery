"""Join-output manifest text: kind='reassembled-safetensors', shards+assets only."""
from .escape import toml_str


def join_manifest_text(plan):
    out = ["schema_version = 1",
           "kind = \"reassembled-safetensors\"",
           f"guarantees = {toml_str(plan.get('guarantees', 'lossless-bytes'))}",
           f"source_fingerprint = {toml_str(plan['source_fingerprint'])}", ""]
    for s in plan["shards"]:
        out += ["[[shards]]", f"path = {toml_str(s['path'])}",
                f"sha256 = {toml_str(s['sha256'])}", f"size = {s['size']}", ""]
    for a in plan["assets"]:
        out += ["[[assets]]", f"path = {toml_str(a['path'])}",
                f"sha256 = {toml_str(a['sha256'])}", f"size = {a['size']}", ""]
    return "\n".join(out)

"""Write moe_split.toml LAST: its presence marks expert-split store complete; join reads it.
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
from pathlib import Path
from pipeline.storage.manifest.escape import toml_str


def emit_manifest(dest, head, files, shas):
    """head: {experts, group, strategy}; files from plan_pieces; shas: {file: {name: [sha]}}."""
    lines = ["# moe_split.toml — expert-aware store map. Join: ms weights moe-join STORE DEST.",
             "schema_version = 1"] + [f"{k} = {toml_str(v) if isinstance(v, str) else v}"
                                      for k, v in head.items()] + [""]
    for file, pieces in files.items():
        for p in pieces:
            lines += ["[[pieces]]", f"file = {toml_str(file)}", f"name = {toml_str(p['name'])}",
                      f"source = {toml_str(p['source'])}",            # original tensor name before destack
                      f"row = {p['row']}",                            # expert row in source; -1 = whole tensor
                      f"dtype = {toml_str(p['dtype'])}", f"source_shape = {p['full_shape']}",
                      f"order = {p['order']}",                        # original tensor order for join
                      f"sha256 = {toml_str(shas[file][p['name']][0])}", ""]
    tmp = Path(dest) / "moe_split.toml.partial"
    tmp.write_text("\n".join(lines))
    tmp.replace(Path(dest) / "moe_split.toml")                       # atomic publish of completion marker

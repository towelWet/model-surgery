# _ancillary_files.py — decide which flat source files copy_ancillary must carry.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pathlib import Path
from pipeline.surgery.shard.tokenizer.discover import discover_assets

INDEX = "model.safetensors.index.json"                        # weight map — caller rewrites it
GENERATION = "generation_config.json"                         # dual-owned: tokenizer manifest lists it,
# but the export contract keeps generation configs in the ancillary set (copy2 is idempotent)


def ancillary_files(src_dir):
    """Relative names of top-level regular files not owned by shards/index/tokenizer copy."""
    source = Path(src_dir).resolve()
    discovered = {str(p) for p in discover_assets(source)} - {GENERATION}
    names = set()
    for entry in source.iterdir():                            # flat only — subdirs owned elsewhere
        if entry.is_symlink() or not entry.is_file():         # regular files only; links never trusted
            continue
        if entry.suffix == ".safetensors" or entry.name == INDEX or entry.name in discovered:
            continue                                          # weights + tokenizer assets copied elsewhere
        names.add(entry.name)
    return names

"""Copy non-weight top-level files (config, tokenizer, templates) into an expert-split store or join output.
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
import shutil
from pathlib import Path

SKIP = {"model.safetensors.index.json", "moe_split.toml"}    # rewritten per layout, never carried stale


def copy_assets(src, dest):
    """Copy top-level regular non-safetensors files; return names of skipped subdirs (reported, not copied)."""
    skipped = []
    for path in sorted(Path(src).iterdir()):
        if path.is_dir():
            skipped.append(path.name)                             # e.g. Reference/ or caches: caller reports
        elif path.name.startswith("._") or path.name == ".DS_Store":
            continue                                              # Finder/exFAT metadata, not model assets
        elif path.suffix != ".safetensors" and path.name not in SKIP and not path.is_symlink():
            shutil.copy2(path, Path(dest) / path.name)            # copy, never move: source stays intact
    return skipped

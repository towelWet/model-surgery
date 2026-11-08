# _copy_tokenizer.py — copy tokenizer files verbatim between checkpoints.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import shutil
from pathlib import Path
from .tokenizer.dir_overlap import check_dirs
from .tokenizer.discover import discover_assets


def copy_tokenizer(src_dir, dest_dir):
    """Copy discovered assets verbatim, preserving nested template paths."""
    source, destination = Path(src_dir).resolve(), Path(dest_dir).resolve()
    check_dirs(source, destination)                            # dir checks before discovery
    assets = discover_assets(source)
    for relative in assets:                                    # validate all targets before any write
        target = destination / relative
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f"Tokenizer destination escapes checkpoint: {relative}")
    for relative in assets:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)

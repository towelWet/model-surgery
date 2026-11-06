# copy_ancillary.py — Copy every flat non-weight non-tokenizer file beside the checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import shutil
from pathlib import Path
from ._ancillary_files import ancillary_files


def copy_ancillary(src_dir: str, dest_dir: str) -> None:
    """Copy discovered ancillary files (configs, remote code, processors) byte-for-byte."""
    destination = Path(dest_dir)
    for name in sorted(ancillary_files(src_dir)):             # sorted: deterministic copy order
        shutil.copy2(Path(src_dir) / name, destination / name)

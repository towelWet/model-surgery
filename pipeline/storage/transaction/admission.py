"""Destination admission shared by split/join: Tau guard + resume-aware existing dirs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # project root for pipeline.* imports
from pipeline.surgery.checkpoint.destination import validate_destination


def admit_destination(destination, sources, resume=False):
    """Validate dest before ANY plan/read/hash/disk work; resume allows existing real dir."""
    dest = validate_destination(destination, sources=sources, allow_existing=resume)
    if resume and (not dest.is_dir() or Path(destination).is_symlink()):
        raise ValueError(f"resume requires existing real directory: {destination}")
    return dest

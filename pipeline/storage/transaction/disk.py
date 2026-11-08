"""Disk preflight: nearest existing destination parent must hold required bytes."""
import shutil
from pathlib import Path


def disk_preflight(destination, required_bytes):
    probe = Path(destination)
    while not probe.exists():                                # walk to nearest existing parent
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < required_bytes:
        raise OSError(f"insufficient disk: need {required_bytes} bytes, have {free} at {probe}")
    return free

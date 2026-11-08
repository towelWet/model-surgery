"""Enumerate every regular checkpoint file; reject links and special files explicitly."""
import os
from pathlib import Path


def checkpoint_files(source):
    source = Path(source)
    files = []
    for base, directories, names in os.walk(source, followlinks=False):
        for name in sorted(directories + names):
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError(f"checkpoint symlinks are unsupported: {path.relative_to(source)}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"checkpoint special file is unsupported: {path.relative_to(source)}")
            files.append(path.relative_to(source).as_posix())
    return sorted(files)

"""Existing destination (file, dir, or dangling symlink) rejected; sentinel preserved."""
import tempfile
from pathlib import Path
from pipeline.surgery.checkpoint.destination import validate_destination


def test_existing_destination_rejected_sentinel_intact():
    with tempfile.TemporaryDirectory() as tmp:
        for kind in ("file", "dir", "link"):
            dest = Path(tmp) / f"out-{kind}-Tau"
            if kind == "file":
                dest.write_bytes(b"sentinel")
            elif kind == "dir":
                dest.mkdir()
                (dest / "sentinel").write_bytes(b"x")
            else:
                dest.symlink_to(Path(tmp) / "gone")         # dangling symlink
            try:
                validate_destination(dest)
            except FileExistsError:
                pass
            else:
                raise AssertionError(f"accepted existing {kind}")
        assert (Path(tmp) / "out-file-Tau").read_bytes() == b"sentinel"
        assert (Path(tmp) / "out-dir-Tau" / "sentinel").read_bytes() == b"x"

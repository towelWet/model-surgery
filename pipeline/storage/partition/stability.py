"""Before commit: source files must keep size+mtime captured at plan; drift rehashes+refuses."""
from ..files.member import safe_member
from ..hashes.digest import file_sha256


def check_source_stable(source, plan, stamps):
    """stamps: {rel: (size, mtime_ns)} captured while hashing each source file."""
    expected = {s["path"]: s["sha256"] for s in plan["shards"]}
    expected.update({a["path"]: a["sha256"] for a in plan["assets"]})
    for rel, (size, mtime) in stamps.items():
        path = safe_member(source, rel)
        stat = path.stat()
        if stat.st_size == size and stat.st_mtime_ns == mtime:
            continue                                        # unchanged; cheap path
        if file_sha256(path) != expected.get(rel):
            raise ValueError(f"source changed mid-copy: {rel}")
    return True

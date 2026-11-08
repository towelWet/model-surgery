"""Per-shard plan entries: raw header record + validated tensors + drift stamps."""
import hashlib
from ..files.member import safe_member
from ..formats.safetensors.coverage import check_shard
from ..formats.safetensors.header import raw_header
from ..hashes.digest import file_sha256
from ..hashes.region import region_sha256


def plan_shards(source, rels, dtype_map):
    shards, tensors, hashes, stamps, names = [], [], [], {}, set()
    for rel in rels:
        path = safe_member(source, rel)
        if not path.is_file():
            raise FileNotFoundError(f"missing shard: {rel}")
        stat = path.stat()
        raw, table, data_size = raw_header(path)
        spans = check_shard(table, data_size, dtype_map)
        shards.append({"path": rel, "header": f"headers/{hashlib.sha256(rel.encode()).hexdigest()}.bin",
                       "sha256": file_sha256(path), "size": stat.st_size,
                       "_header_bytes": 8 + len(raw), "_mtime": stat.st_mtime_ns})
        for (begin, end), name in spans:
            if name in names:
                raise ValueError(f"duplicate tensor name across shards: {name}")
            names.add(name)
            entry = table[name]
            tensors.append({"name": name, "dtype": entry["dtype"], "shape": entry["shape"],
                            "shard": rel, "offsets": [begin, end],
                            "file": f"tensors/{hashlib.sha256(name.encode()).hexdigest()}.safetensors",
                            "sha256": region_sha256(path, stat.st_size - data_size + begin, end - begin),
                            "nbytes": end - begin})
        hashes.append((shards[-1]["sha256"], rel))
        stamps[rel] = (stat.st_size, stat.st_mtime_ns)
    return shards, tensors, hashes, stamps

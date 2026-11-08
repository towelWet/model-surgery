"""split_experts: MoE checkpoint -> expert-aware store (strategy-chosen files + HF index + moe_split.toml).
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
Guarantee: every tensor byte preserved (per-piece sha256; moe-join restores stacked layout).
Not a quality/RAM claim: on-demand expert residency is the consumer runtime's job.
"""
import shutil
from pathlib import Path
from pipeline.storage.experts.copy_assets import copy_assets
from pipeline.storage.experts.emit_manifest import emit_manifest
from pipeline.storage.experts.expert_count import expert_count
from pipeline.storage.experts.load.layout import load_layout
from pipeline.storage.experts.load.strategies import load_strategies
from pipeline.storage.experts.plan_pieces import plan_pieces
from pipeline.storage.experts.read_tensors import read_tensors
from pipeline.storage.experts.write.index import write_index
from pipeline.storage.experts.write.multi import write_multi


def split_experts(src, dest, strategy="expert", group=1, dry_run=False):
    """Return {experts, strategy, group, files:{file: bytes}, skipped}; dry_run writes nothing."""
    src, dest = Path(src).resolve(), Path(dest).resolve()
    if dest == src or src in dest.parents or dest in src.parents:
        raise ValueError("destination must not overlap source")      # never write inside the checkpoint
    lay, strategies = load_layout(), load_strategies()
    if strategy not in strategies or group < 1:
        raise ValueError(f"strategy must be one of {sorted(strategies)} and --group >= 1")
    experts = expert_count(src, lay)
    files = plan_pieces(read_tensors(src), experts, lay, strategies[strategy]["run"], group)
    sizes = {f: sum(r[2] for p in ps for r in p["regions"]) for f, ps in files.items()}
    head = {"experts": experts, "strategy": strategy, "group": group}
    if dry_run:
        return {**head, "files": sizes, "skipped": []}
    if shutil.disk_usage(dest.parent).free < sum(sizes.values()) * 1.02:   # 2% slack: headers + assets
        raise ValueError(f"{dest.parent}: not enough free space for {sum(sizes.values())} bytes")
    dest.mkdir()                                                     # refuses existing dest: no overwrite
    shas = {f: write_multi(dest / f, ps) for f, ps in files.items()}
    skipped = copy_assets(src, dest)
    write_index(dest, files)
    emit_manifest(dest, head, files, shas)                           # LAST: marks store complete
    return {**head, "files": sizes, "skipped": skipped}

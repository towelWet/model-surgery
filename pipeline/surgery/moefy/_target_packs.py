# _target_packs.py — discover dense->MoE target packs (targets/*.toml), keyed by dense architecture.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Same hot-load idea as tensor-array ArchRegistrar: support a new arch = drop one pack file.
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent / "targets"


def target_packs(folder=HERE):
    """Return {dense_arch: pack}; two packs claiming one arch = hard error, never silent override."""
    packs = {}
    for path in sorted(Path(folder).glob("*.toml")):
        pack = {**tomllib.loads(path.read_text()), "pack": path.name}
        arch = pack["dense_arch"]
        if arch in packs:
            raise ValueError(f"{arch} claimed by {packs[arch]['pack']} and {path.name}")
        packs[arch] = pack
    return packs

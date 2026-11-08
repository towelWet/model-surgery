# ../moe/partition/run.py — CLI: build expert partitions per layer. Geometry is explicit caller policy.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Input : <stats>/L<digits>/{neuron.npy, sample.npy}      (sparse ids fine: L02, L07)
# Output: <stats>/partition/<mode>/L<digits>.npz          (modes = modes/**/manifest.py)
# Usage: python3 surgery/moe/partition/run.py <stats_dir> --experts E --expert-width W
#        --shared-width S --group-size G [--layers 0-55]
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.moe.partition.plan.arguments import parse_args
from pipeline.surgery.moe.partition.plan.selection import discover_layers
from pipeline.surgery.moe.partition.plan.validation import validate
from pipeline.surgery.moe.partition.modes.discovery import discover_modes
from pipeline.surgery.moe.partition.output.layer import run_layer

args = parse_args()
items = discover_layers(args.stats_dir, args.layers)
modes = discover_modes()
if not modes:
    raise SystemExit("moe_partition: no modes discovered under modes/**/manifest.py")
p = SimpleNamespace(experts=args.experts, expert_width=args.expert_width,
                    shared_width=args.shared_width, group_size=args.group_size, width=None)
widths = validate(args.stats_dir, items, modes, p)            # whole input set checked first

for name, man in modes.items():
    for _, dname in items:
        p.width = widths[dname]
        run_layer(args.stats_dir, dname, name, man, p)
    print(f"  {name}: {len(items)} layers partitioned", flush=True)

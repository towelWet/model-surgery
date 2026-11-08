# Pipeline stages — public entry points

Run from the directory containing SKILL.md. Per-model work is private and optional.
Never skip profiling before choosing a structural edit.

| Stage | Entry | Status |
|---|---|---|
| Inspect | `pipeline/cli/ms.py inspect MODEL` | Header-only checkpoint geometry and shard checks |
| Discover | `pipeline/cli/ms.py list --search TEXT` | Offline tool inventory |
| Profile | `pipeline/profile/expert/routing_profile.py` | tensor-array HOPE routing statistics |
| Profile | `pipeline/profile/block/distance.py` | Blocked: per-layer residual instrumentation needed |
| Profile | `pipeline/profile/mlp/stats.py` | Blocked: MLP taps needed |
| Surgery | `pipeline/surgery/depth/prune.py` | Remove selected decoder layers |
| Surgery | `pipeline/surgery/moe/partition/run.py` | Partition recorded neuron statistics |
| Eval | `pipeline/cli/ms.py cases` | Offline case validation |
| Eval | `pipeline/eval/run/all.py` | Diagnostic scores; not promotion-grade proof |
| Gate | `pipeline/cli/ms.py gate` | Rejects diagnostic/incomparable evidence |
| Heal | `pipeline/heal/tool/distill.py` | Requires tensor-array-train; not validated here |
| Export | `pipeline/export/mlx/convert/core.py` | Mixed-bit export tools |

See `README.md` for the full inventory and known blocked interfaces.
See `db/surgery/methodology/stages-annex.md` for current command examples.

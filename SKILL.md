---
name: model-surgery
description: Restructure trained LLMs without retraining — depth/width/expert pruning, dense→MoE, TIES-DARE merge, abliteration, neuron edits, KD heal via tensor-array-train, mixed-bit export, eval suites + promotion gate. CLI-only (`ms` doctor/eval/gate/list). Use on /model-surgery, pruning/MoE/heal/quant questions, or Tau surgery campaigns. tensor-array only; never mlx_lm.
---
# /model-surgery — Model Surgery & Restructuring

> **⛔ ∞ CRACKME DBV = MoE ONLY:** trial models, eval candidates, baselines and surgery targets must be MoE (config has experts). Dense models (Qwen3.8-27B, Tau-56L, Bonsai 27B) never.
> **⛔ ∞ NO MLX_LM — EVER:** `mlx_lm` / `mlx_vlm` (incl. `mlx_lm.server`, Bonsai-demo `.venv-vlm`) are never allowed — not for serving, eval, baselines, cross-checks, calibration, pruning, training or fallback. tensor-array is the only inference stack; fix gaps in tensor-array. Scan every command for `mlx_lm`/`mlx_vlm` before running.

Restructure a **trained model** without full retraining: structural prune, MoE
conversion/compression, distillation/heal. The skill **is** the project at
the directory containing this file (`.grit/` tracks state). Optional private locations
live in local.toml; that file is structurally excluded from publication.

Current checkpoint I/O: safetensors. Direct GGUF editing/export is not implemented.
Offline metadata/evidence commands do not require the inference runtime.

## Per-model work layout (version-and-lanes layout — user rule 2026-11-02)

Every surgery target lives at `db/<family>/<major>_<minor>/<size>/` (product), then a
**version folder**, then the standard lanes. Never invent layer names (`runs/`, `intact/`,
`1-setup/`, `steer-*`): use the version-and-lanes layout shown below.

```
db/qwen/3_6/35B-A3B/                ← product: current.toml, INDEX.toml, aliases.toml, profile/
├── 1/                              ← version 1 = one DIRECTION (prune-and-heal on intact SLAB)
│   ├── 1-original/model            ← pristine base, never touched
│   ├── 2-test/<N>/                 ← each attempt (rerun, hope224, router-bf16 …); N+1, never overwrite
│   ├── 3-final/                    ← promoted winner only, flat
│   └── diary/ plans/ scripts/ PROCESS.md
└── 2/                              ← NEW DIRECTION or a different surgery program, same lanes
```

| Change | Goes in |
|---|---|
| another build/rerun of same idea | `<version>/2-test/<N+1>/` |
| different direction / different surgery work | new version `<N+1>/` |

Heavy folders follow the full-drive alias rule: real data on roomy drive, symlink at this
path, row in a private per-model aliases.toml. Live example: `db/qwen/3_6/35B-A3B/` (Grim Golem).

## Read order

`SKILL.md` → private product state (when present) → lane doc below.
Code: `README.md` (script tables). Intended workflow: `pipeline/profile/` → `pipeline/surgery/` →
`pipeline/heal/` → `pipeline/export/`, gated by `pipeline/eval/`. Profile-stage per-layer/MLP/layer-skip
instrumentation is unavailable, so dependent campaigns are blocked — not bypassed; offline
inventory/inspection and standalone commands remain usable. Passing tests are not real-model proof.

## CLI (`ms`)

| Command | Does |
|---|---|
| `python3 pipeline/cli/ms.py list [stage]` | runnable scripts per stage, read from disk |
| `python3 pipeline/cli/ms.py doctor [check]` | imports · size · collisions · mlx_lm · shadowing · litter · docs |
| `python3 pipeline/cli/ms.py eval --out-dir D --model L` | tool/code/rev (+kl) suites → `D/<suite>.toml` |
| `python3 pipeline/cli/ms.py gate --base B --cand C` | rules in `pipeline/eval/gate.toml`; writes verdict.toml into candidate dir; exit 1 = blocked |

Promotion: eval base model once, eval each `2-test/<N>/`, gate; only a passing verdict may be
copied into `3-final/` (copy stays a human step — gate never moves models).
Run `ms doctor` after any code change; it must stay clean. Disabled profiling is not functionality.
`ms inspect MODEL` reads checkpoint metadata/shard bounds without loading weights.
`ms cases` validates eval data offline; `ms list --search TEXT` filters commands.
Current eval reports are diagnostic-only and cannot pass promotion. Fingerprints/counts
must match; missing evidence, non-finite metrics and empty rules block promotion.

## Known block: in-process profilers

`block/distance.py`, `mlp_stats.py`, `block/ablation.py --score`, `layer/ablation_serve.py`,
`neuron/probe.py` ran forward passes through `mlx_lm` (banned). `pipeline/profile/_mlx_load.py`
now refuses. tensor-array lacks per-layer hidden dump, layer skip, MLP taps (checked 2026-11-06).

Ported: `expert/routing_profile.py` — serve with `TARR_EMBEDDINGS=1 TARR_HOPE_STATS=<path>`,
run with `--stats <path>`; hits + co-activation read from tensor-array's HOPE file.

## Lane docs

| Topic | File |
|---|---|
| Stage order + script table | `db/surgery/methodology/stages.md` |
| Stage command examples | `db/surgery/methodology/stages-annex.md` |
| Naming, memory, eval, heal chain | `db/surgery/methodology/discipline.md` |
| Profile + packed-GEMM gates | `db/surgery/methodology/gate.md` · `db/surgery/methodology/gate-annex.md` |
| Legitimacy checklist + arch traps | `db/surgery/methodology/legitimacy.md` |
| Papers + Reference/ catalog | `db/surgery/methodology/references.md` |
| MoE conversion & compression | `db/surgery/techniques/conversion-moe.md` |
| Prune, distill, active-param speed | `db/surgery/techniques/pruning-distill-speed.md` |
| Decode ceilings by active params | `db/surgery/techniques/active-b-decode-ceilings.md` |
| MoE expert surgery — 2026 methods | `db/surgery/moe/INDEX.toml` |
| Balanced heal (production) | `pipeline/heal/BALANCED_HEAL.md` |
| Code/rev + KD heal | `pipeline/heal/CODE_REV_TRAIN.md` |
| Gate shell tools | `pipeline/profile/preflight.sh` · `pipeline/profile/benchmark/report.sh` · `pipeline/profile/benchmark/run.sh <config.toml>` · `pipeline/profile/perf-monitor.sh` · `pipeline/profile/perf-log-view.py` |

Missing since before 2026-11-06 (cited by old docs, not on disk): OPERATOR, MODEL_COMPARE,
OVERNIGHT_20260921, OVERNIGHT_GUIDE, RETROSPECTIVE_20260921, KNOWN_ISSUES, BONSAI_GATE,
BONSAI_PERF_ANALYSIS, MORNING_20260921(+ANNEX), SURGERY_20260921(+ANNEX), SKILL_TECHNIQUES_PRUNE_ANNEX.

Local cross-project research locations can be recorded in local.toml; they do not ship.

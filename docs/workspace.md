# Per-model work layout (version-and-lanes layout — user rule 2026-11-02)

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


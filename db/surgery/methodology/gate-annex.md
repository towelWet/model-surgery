# Profiling gate annex — use_metal config

Moved from `db/surgery/methodology/gate.md` (50-line cap). Gate tables stay there.

## use_metal config (2026-09-21 03:22)

| Check | Result | Ready? |
|---|---|---|
| `ServerConfig.use_metal` | **false** default | yes |
| Flip | env `TARR_USE_METAL=1` **or** TOML `use_metal = true` | yes |
| Builds | lib + server `-j2` **pass** (03:22 EDT, then 03:36:38) | yes |

Config has **no** `use_metal` key (stays false). `qmm_use_metal()` = env OR config.

`use_metal=true` enables Metal JIT for **1-bit only**; ternary still CPU.
Metal JIT API/scope: `db/surgery/methodology/gate.md`. Morning: MORNING_20260921.md [missing: not on disk 2026-11-06].

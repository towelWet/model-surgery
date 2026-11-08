# Profiling gate — what to cut

**Never commit a structural cut based only on model size.**
Every cut competes on Δquality / Δactive-parameter.
Run `pipeline/profile/block/distance.py` first; cut what it ranks, not what looks tidy.

Corollary — local-only history: db/rejected/weight-norm-similarity/similarity.py identical `1.92e+16` all 16 groups (ranked nothing).
`pipeline/profile/block/distance.py` is currently blocked pending tensor-array per-layer instrumentation.
The measurements below are historical records, not a current working profiler.

## Measured — Qwen3.8-27B, 64L, 3-bit, 26 calibration prompts (historical — profiler currently blocked)

Angular distance (PruneMe), ascending = most redundant:

| 16-layer removal | Σ group distance | Outcome |
|---|---|---|
| `4-15,40-43` | 0.336 | best, non-contiguous (Tau-48L — rejected) |
| `28-43` | 0.459 | best contiguous |
| `24-39` | 0.480 | blind cut → 20% tool acc |

Layers `0-3` score 0.4818, `60-63` score 0.3172 — never cut either end.
Cuts must align to `full_attention_interval` (4) or the
(linear, linear, linear, full) group pattern breaks.

**Production cut:** `"8-15"` → Tau-56L. Log:
a private per-model surgery log (not shipped).

## Packed GEMM readiness (2026-09-21)

| Check | Result | Ready? |
|---|---|---|
| `test-quant-affine-1bit` (bits=1, K=5120) | **pass** (host) | yes |
| `test-quant-packed-ternary` (K=5120) | **pass** (0xAA +1 / 0x00 -1) | yes |
| Metal JIT | in binary **06:42 EDT**; **not run** | no |
| Metal shader syntax (`xcrun metal -c`) | 1-bit OK, ternary OK, exit 0. Runtime dispatch untested (morning) | syntax yes |

Host CPU default. Overnight: OVERNIGHT_20260921.md [missing: not on disk 2026-11-06]. Config: `db/surgery/methodology/gate-annex.md`.

## Metal JIT gate (2026-09-21 06:42)

| Check | Result | Ready? |
|---|---|---|
| API | `mx::fast::metal_kernel` (not `compile_kernel`) | yes |
| `mx::eval` | **none** on Metal result (lazy graph) | yes |
| Scope | 1-bit + ternary wired; GPU morning | yes |
| Bonsai dispatch shapes | VERIFIED (gate_proj out=17408, in=5120, gs=128) (2026-09-21 build-time check, not re-run). | yes |
| Binary | server `-j2` **pass** (06:42 EDT, 2.9M) | yes |

`use_metal=true` / `TARR_USE_METAL=1` → Metal JIT 1-bit+ternary. GPU not executed tonight.
`use_metal` in `/api/status`+`/health`: **yes** (06:42 build, `qmm_use_metal()` = env OR config).

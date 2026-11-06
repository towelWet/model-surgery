# Discipline — naming, memory, eval, heal order

## Heal chain (Phase 2)

```
gen_corpus  →  cache_teacher_logits  →  train_lora_distill  →  fuse_adapter
     ↓                    ↓                      ↓
 distill_corpus.jsonl   kd_top32.npz      adapters/  →  …-Tau-…-healed
```

Unload 27B before teacher cache; restore Tau-56L on `:8765` after. Never hold both in RAM — cache is two-pass.
Production: `mlx_lm lora`/`fuse` (`heal/BALANCED_HEAL.md`); local-only history: db/rejected/mlx-lm-heal/train_lora_distill.py is `mlx_lm lora`, not tensor-array-train.

`<spec>` accepts non-contiguous removals: `"4-15,40-43"`.

## NAMING (non-negotiable)

Every checkpoint — prune, merge, heal, fuse, requant, abliterate — **must** include `Tau`
in the directory or filename.
Examples: `Qwen3.8-27B-Tau-56L-3bit`, `Qwen3.5-4B-Tau-Huihui-Qwopus-TIES-4bit`.
Do not replace existing `Tau-56L` names. Do not rename historical rows in `db/<family>/<ver>/<size>/results.toml`.

## MEMORY

Never run a prune and a resident model server together.
`pipeline/surgery/depth/prune.py` peaks ~9 GB; resident 27B 3-bit server holds 8.5–11 GB.

Fully unload (not just stop) any inference server before surgery — keep-alive may restart a plain stop.
Don't double-load (no `:8766` while `:8765` is up; no overnight swap).

## EVAL

Two harness bugs silently invalidated earlier scores:
| Requirement | Why |
|---|---|
| `enable_thinking` at **top level** | tensor-array drops `chat_template_kwargs`; template keeps open thinking branch |
| `temperature: 0.0` | default sampling made identical requests non-deterministic for tool calls |

Baseline the **unpruned** parent before cutting. Post-surgery numbers need that reference.
`eval_every=8` applies at layer boundaries within each forward pass; separate `async_eval` per token overlaps token decode.

Production gates: OPERATOR.md [missing: not on disk 2026-11-06].
Run `profile/preflight.sh` before bootout (exit 0 = go).
Preflight: run before bootout; fails on missing server binary, benchmark entrypoints, or eval port.
`profile/benchmark/run.sh <config.toml>` smoke-checks a served model; `profile/benchmark/report.sh` runs preflight.
Crackme: 6/6 (b64, rev, const, dict, plain, assign-retroactive). ask.py strips quotes/backticks/prefix/first-word.

## DOCS

`.md` slices ≤50 (this file included; many companions already near 50). Split a companion rather than grow. Historical exceptions: local-only history: db/qwen/3_8/27B/surgery.md, OVERNIGHT_20260921.md [missing: not on disk 2026-11-06].

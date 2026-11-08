# Architecture compatibility notes

## `model_type` naming traps

Reflects MoE **infrastructure**, not base family.

| model_type | Meaning |
|---|---|
| `qwen3_5` | Qwen3.8-27B dense (Qwen3.5 arch internals) |
| `qwen3_5_moe` | Qwen3.5-MoE routing on some base |
| `qwen3_moe` | Qwen3-30B-A3B / 235B-A22B family |

Verify via model tree, not `model_type` string.

## tensor-array arch support

```bash
rg "Qwen3_5Moe\|Qwen3Moe\|MoeFor" <tensor-array-src>/arch/ --type cpp
```

Register target arch before pulling weights. Tau serve arch:
`TARR_ARCH=Qwen3_5ForConditionalGeneration` — see OPERATOR.md [missing: not on disk 2026-11-06].


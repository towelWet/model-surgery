# Dense-to-MoE (`pipeline/surgery/moefy/`)

Turns dense Qwen3 into Qwen3MoE that runtime loads as-is.

**Experts.** MLP neurons that fire on same tokens are grouped into equal-size experts
(balanced k-means on activation patterns, MoEfication-style).

**Routers.** Each layer's router is ridge-regressed from real MLP inputs to measured
expert output size. Logits are kept tiny so chosen experts weigh ~1/K; `down_proj` is
scaled by K so selected experts add up like dense neurons.

**Non-MLP tensors** are copied from same model quantized by runtime's own quantizer,
so dense-vs-MoE comparisons isolate MLP change.

## Steps

```sh
# 1. serve quantized dense model with probe gate on, then capture MLP inputs
python3 pipeline/profile/moefy/capture.py CALIB_DIR --pieces 64
# 2. convert (expert width must be multiple of quant group size)
python3 pipeline/surgery/moefy/entry.py BF16_DIR QUANT_DIR CALIB_DIR DEST --experts 8 --topk 4
# 3. heal: dense teacher cache + LoRA on attention and routers, then fuse
python3 pipeline/profile/moefy/heal_rows.py rows.jsonl
TRAIN cache-teacher --model QUANT_DIR --data rows.jsonl --out CACHE_DIR
TRAIN train --model DEST --data rows.jsonl --teacher-cache CACHE_DIR --out LORA --steps 256 --lr 2e-4
TRAIN fuse --model DEST --adapter LORA --out HEALED
# 4. score unseen text (serve each model with probe gate on)
python3 pipeline/profile/moefy/nll.py
```

`moefy_report.json` in DEST lists per-layer held-out output error for router,
oracle (true top-K) and random choice.

## Measured result

Model: Qwen3-0.6B, 8-bit. Calibration: 16k tokens of stdlib code + man pages.
Score: mean NLL on 32 pieces never used for routers or healing (lower is better).

| Variant | MLP compute | Held-out NLL |
|---|---|---|
| Dense (8-bit) | 100% | 2.4401 |
| 8 experts, all active | 100% | 2.4409 |
| 8 experts, 4 active | 50% | 4.8021 |
| 8 experts, 4 active, healed (attention + routers) | 50% | 3.5390 |
| 8 experts, 4 active, healed (+ routed-expert LoRA) | 50% | 3.4095 |
| 8 experts, 2 active | 25% | 11.2728 |
| 16 experts, 8 active | 50% | 4.8227 |
| 48 experts, 24 active | 50% | 5.6472 |

Re-measured 2026-11-07 after runtime tokenizer fix ([pruning](pruning.md#runtime-tokenizer-fix-2026-11-07));
older numbers in history used broken pre-tokenizer.
All-experts output matches dense: conversion and loader are exact.
Router output error 0.51 vs oracle 0.48 vs random 0.68 (8 experts, 4 active).
`--expert-lora` (trainer) adds LoRA to stacked routed experts: 3.5390 → 3.4095.

Limits: one small model; calibration, heal and eval text share a domain (code + man
pages). Evidence that pipeline works, not a
general quality guarantee.

### Qwen3.5 4B hybrid (`moefy/hybrid/entry.py`, 8 experts, no healing)

| Active experts | Held-out NLL |
|---|---|
| dense bf16 | 1.8099 |
| 8 | 1.8205 |
| 4 | 2.4514 |
| 2 | 3.0297 |

All-active matches dense closely; fewer active experts need healing, as on Qwen3-0.6B above.

### Llama 3.2 3B → Mixtral layout (`moefy/entry.py`, 8 experts, 4-bit, no healing)

| Active experts | Held-out NLL |
|---|---|
| dense 4-bit | 2.2709 |
| 8 | 2.2783 |
| 4 | 4.5378 |

Router output error (mean of 28 layers, top-4): router 0.560, oracle 0.515, random 0.685.
Before 2026-11-08 Llama forward returned last-token logits only, so probe NLL failed; fixed in
runtime (all positions when probe/training asks). Conversion reads base non-MLP tensors on CPU
(`moefy/_non_mlp.py`): left lazy, save hit GPU watchdog on top-4.

## Arch-agnostic entry (`ms moeify`)

```sh
python3 pipeline/cli/ms.py moeify plan MODEL_DIR            # read-only: already MoE / pack / unsupported
python3 pipeline/cli/ms.py moeify convert BF16 QUANT CALIB DEST --experts 8 --topk 8
python3 pipeline/cli/ms.py moeify split MOE_DIR STORE --dry-run
```

Dense arch → converter comes from target packs `pipeline/surgery/moefy/targets/*.toml`
(`dense_arch`, `converter`, `block`, `config_writer`, `moe_arch`, `status`).
Previously hardcoded `TARGETS` dict in `pipeline/surgery/moefy/moe/_target.py`; new arch = one pack file
plus, if its MLP shape differs, its own converter dir. Split side: [experts](storage/experts.md).

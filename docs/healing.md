# Heal (`pipeline/heal/`)

| Script | What it does |
|---|---|
| `gen/corpus/core.py` | Build distillation JSONL |
| `cache_teacher_logits.py` | Top-32 teacher logprob harvest via tensor-array |
| `tool/distill.py` | Tool-routing LoRA via tensor-array-train (CE + top-32 KL) |
| `general_distill.py` | General corpus pass, lower LR, after tool_distill |
| `fuse_adapter.py` | Fuse LoRA adapter via `tensor-array-train fuse` |
| `run_heal.sh` | Corpus → cache → tool_distill → fuse |

## Measured result

Model: 4B hybrid, 4-bit; layers 20–23 removed (32 → 28).
Data: teacher-written answers; 8 held-out rows never trained on.
Heal: LoRA rank 16, 64 steps, lr 1e-4, then fused.
Score: mean masked cross-entropy on held-out answers (lower is better).

| Model | Held-out loss |
|---|---|
| Teacher (unpruned) | 1.8554 |
| Pruned, not healed | 2.7598 |
| Pruned + healed | 2.0667 |

Healing recovered 77% of loss gap from pruning.
One model, one cut, 8 rows: evidence that loop works, not a quality guarantee.

Trainer fixes this result depended on: fp32 per-position loss normalizer,
LoRA tensors passed through gradient checkpoints, fuse multiplying
`A @ B` (not `B @ A`) with each layer's true input width, one GPU eval per fused layer.

## Trainer modes (`tensor-array-train train`)

| Flag | Trains | Output |
|---|---|---|
| (default) | LoRA on attention, dense MLP, MoE router + shared expert | adapter → `fuse` |
| `--expert-lora` | also LoRA per routed expert (Qwen3-MoE `switch_mlp` stacks, via `gather_mm`) | adapter → `fuse` |
| `--full` | decoder linear weights themselves, fp32 master copies, AdamW | full checkpoint |
| `--full --keep-zeros` | same, zeros of source stay zero (gradient-masked pruning) | full checkpoint |
| `--full --l1 F` / `--group-lasso F` | adds sparsity penalty (mean \|W\| / mean row L2 per tensor, × F) | full checkpoint |

Sparsity penalty, Qwen3-0.6B, full FT 64 steps lr 5e-5 (held-out NLL: after training / after 50 % magnitude MLP prune):

| Penalty | Trained | Pruned |
|---|---|---|
| none | 2.3606 | 3.4002 |
| `--l1 2` | 2.3585 | 3.4069 |
| `--group-lasso 2` | 2.3582 | 3.3898 |
| `--l1 20` | 2.3579 | 3.3921 |
| `--group-lasso 20` | 2.3222 | 3.2843 |

Only group lasso at ×20 helps pruning (−0.116). Penalty is a loss term under Adam, so per-step
weight motion stays bounded by learning rate; a proximal soft-threshold step is next if L1 must bite.

`--teacher-cache DIR` adds top-32 KD to any mode, so `--full` with a cache is full distillation.
Weights are paged in before step 0 (lazy reads inside first GPU step tripped watchdog; 5/5 cold
runs pass after fix).

### Sparse model heal — Qwen3-0.6B Wanda 50 % all linears (held-out NLL, dense 2.4346)

| Heal | Zeros kept | NLL |
|---|---|---|
| none | 50 % | 2.8469 |
| LoRA r16 + fuse | 0 % | 2.4926 |
| LoRA r16 + fuse + `sparse/remask/core.py` | 50 % | 2.5398 |
| `--full --keep-zeros` (lr 1e-5) | 50 %, same mask | 2.5011 |

All 64 steps on same 64 rows with same teacher cache. Mask equality checked tensor by tensor.

### MoE heal — Qwen3-0.6B → 8 experts, top-4 (moefy), held-out NLL (dense 2.4346)

| Heal (64 steps, r16, KD) | LoRA tensors | NLL |
|---|---|---|
| none | — | 4.8021 |
| attention + router | 140 | 3.5390 |
| attention + router + routed experts (`--expert-lora`) | 224 | 3.4095 |

Fuse runs on CPU: stacked expert dequantize/requantize timed out GPU watchdog.

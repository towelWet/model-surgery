# Surgery (`pipeline/surgery/`)

| Script | What it does |
|---|---|
| `depth/prune.py` | Remove transformer blocks by index or ablation rank |
| `width/prune.py` | Slice dense FFN intermediate_size (bf16 and any bit-width) |
| `mlp/only_prune.py` | Zero MLP weights (incl. quant scales/biases) in selected layers |
| `expert/prune/core.py` | Drop weakest MoE experts (stacked `switch_mlp` or per-expert keys) |
| `moefy/entry.py` | Dense Qwen3 → Qwen3MoE: co-activation experts + fitted routers ([moefy](moefy.md)) |
| `moefy/hybrid/entry.py` | Dense Qwen3.5/3.6 → Qwen3.5-MoE: shared expert + routed experts ([moefy](moefy.md)) |
| `dequant/entry.py` | MLX affine checkpoint → bf16 (start point for structural edits) |
| `wquant/entry.py --method rtn\|awq\|gptq --bits 4 --group 64 --calib DIR` | bf16 → MLX affine; AWQ scales channels by captured inputs, GPTQ corrects error column by column from input Hessian |
| `moe/partition/run.py` | Expert partitions per layer; geometry via required flags `--experts --expert-width --shared-width --group-size`; modes from `modes/**/manifest.py` |
| `merge/ties/core.py` | Merge fine-tunes of one base: 13 methods ([merging](merging.md)) |
| `passthrough/entry.py` | Stack layer slices from compatible models (`--slices "0:0-15,1:8-23"`) |
| `soup/greedy/core.py` | Greedy soup: add fine-tunes while held-out score improves |
| `heads/`, `heads_linear/`, `hidden/`, `sparse/`, `lowrank/`, `vocab/` | Structured, unstructured, low-rank and vocabulary pruning ([pruning](pruning.md)) |
| `abliterate/core.py` | Project out refusal direction |
| `neuron/edit.py` | Patch neuron output columns via activation contrast |
| `neuron/recipe/core.py` | Steering recipe from target vs baseline prompts |

Weight quantization, Qwen3-0.6B held-out NLL (2026-11-08, calibration = `pipeline/profile/moefy/capture.py` taps):

| Model | NLL |
|---|---|
| bf16 | 2.4346 |
| 8-bit runtime quantizer | 2.4401 |
| 4-bit g64 RTN | 2.5491 |
| 4-bit g64 AWQ | 2.5233 |
| 4-bit g64 GPTQ | 2.4686 |

GPTQ closes 70 % of 4-bit RTN gap to bf16.

Neuron edits and output-space projections dequantize MLX affine triples using explicit
global/per-module metadata and regenerate packed weights, scales and biases together.
Unknown formats, missing metadata and incompatible dimensions fail explicitly.
Discovery accepts one indexed block namespace; ambiguous text/vision stacks require
a model-specific adapter and are refused. Activation contrast uses signed means.

Neuron editing, projection, expert pruning and DARE write logical groups incrementally.
The input cache holds at most one shard per source; a single large shard/matrix still
sets the working-set requirement. This is not runtime expert streaming or a hard RAM cap.

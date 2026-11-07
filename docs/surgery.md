# Surgery (`pipeline/surgery/`)

| Script | What it does |
|---|---|
| `depth/prune.py` | Remove transformer blocks by index or ablation rank |
| `width_prune.py` | Slice dense FFN intermediate_size (bf16 and any bit-width) |
| `mlp/only_prune.py` | Zero MLP weights (incl. quant scales/biases) in selected layers |
| `expert/prune.py` | Drop weakest MoE experts (stacked `switch_mlp` or per-expert keys) |
| `moefy/entry.py` | Disabled: trained routers + loader contract required |
| `moe_partition/run.py` | Expert partitions per layer; geometry via required flags `--experts --expert-width --shared-width --group-size`; modes from `modes/**/manifest.py` |
| `merge_ties.py` | DARE task arithmetic; TIES is not implemented |
| `abliterate.py` | Project out refusal direction |
| `neuron/edit.py` | Patch neuron output columns via activation contrast |
| `neuron/recipe.py` | Steering recipe from target vs baseline prompts |

Neuron edits and output-space projections dequantize MLX affine triples using explicit
global/per-module metadata and regenerate packed weights, scales and biases together.
Unknown formats, missing metadata and incompatible dimensions fail explicitly.
Discovery accepts one indexed block namespace; ambiguous text/vision stacks require
a model-specific adapter and are refused. Activation contrast uses signed means.

Neuron editing, projection, expert pruning and DARE write logical groups incrementally.
The input cache holds at most one shard per source; a single large shard/matrix still
sets the working-set requirement. This is not runtime expert streaming or a hard RAM cap.

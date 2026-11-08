# Expert-aware split (`pipeline/storage/experts/`)

One logical MoE checkpoint, many physical files, plus standard HF index.

```sh
python3 pipeline/cli/ms.py moeify split MOE_DIR STORE [--by expert|layer] [--group G] [--dry-run]
python3 pipeline/cli/ms.py weights moe-join STORE OUT
```

## Dynamic parts (no model family in code)

| Concern | Lives in | Extend by |
|---|---|---|
| tensor-name grammar, config keys, file names | `pipeline/storage/experts/layouts/default.toml` | drop another layouts TOML |
| grouping strategy | `pipeline/storage/experts/strategies/expert.py`, `…/layer.py` | drop one strategies file with `STRATEGY` |

Layout TOMLs merge: lists union, conflicting scalars refuse. Grammar is structural
(dot components validated against E), same approach as tensor-array
`infer/moe/stream/expert/key-parse.cpp`.

## Output layout

Stacked `[E, ...]` tensors (quantized weight/scales/biases alike) are destacked into
per-expert `<name>.eNNN` tensors without leading E dim — tensor-array slab-repack
convention its expert streamer reads. Inside each file: resident tensors first, then
layer → expert → tensor, so every expert is one contiguous byte span (tensor-array
`src/infer/moe/stream/slab/plan.cpp` coalesces that into one pread). HF per-expert tensors move whole.
Shared experts, routers, attention, norms, embeddings stay resident.

File names default to `model-*.safetensors` because tensor-array loads only
`model-*` shards when an index exists; `<store>/moe_split.toml` (written last =
completion marker) records source name, expert row and sha256 of every piece.

## Guarantee

Tensor bytes only: `moe-join` restacks rows in original tensor order and refuses
output on any piece sha256 mismatch. No inference, quality or RAM claim.

## Verified (2026-11-07)

Synthetic tests: `tests/storage/moe_split_{stacked,single,layer,refusals}.py`,
`tests/storage/moe/layout_merge.py`.
Real headers, dry-run only: Qwen3-Coder-30B-A3B-4bit (128 experts, 432 stacked
tensors = 48 layers × 9) and Qwen3.6-35B-A3B-AIMER-64e (360 = 40 × 9), zero
misclassified names. Not yet run: real-byte split+join, and tensor-array loading a
split store (drives near full).

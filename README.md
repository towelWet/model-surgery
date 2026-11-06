# Model Surgery

Restructure trained LLMs without full retraining. Prune layers, experts or FFN width,
convert dense FFNs to MoE, merge checkpoints, abliterate refusal directions, then heal
with knowledge distillation. Current weight operations use MLX and safetensors;
the configured local inference runtime is tensor-array (Apple Silicon).

**Formats:** direct GGUF editing/export is not implemented. Offline inventory and
evidence commands do not require inference; no Linux/Windows weight-editing claim is made.

![ms doctor, cases and list output](docs/figs/cli-checks.png)

## Pipeline

```
pipeline/profile/  →  pipeline/surgery/  →  pipeline/heal/  →  pipeline/export/
                                  pipeline/eval/  (gate before promotion)
```

Intended workflow order: profile → surgery → heal → export. Per-layer, MLP and
layer-skip instrumentation required by the profile stage is currently unavailable
(below), so dependent campaigns are blocked — not bypassed. Offline
inventory/inspection and standalone commands needing no instrumentation remain
usable. Passing tests are not real-model proof. All code lives under `pipeline/`;
per-model work lives under `db/`.

## CLI

```
python3 pipeline/cli/ms.py              # list commands
python3 pipeline/cli/ms.py list         # every runnable script per stage, from disk
python3 pipeline/cli/ms.py list --search prune  # filter paths and descriptions
python3 pipeline/cli/ms.py inspect MODEL       # metadata + shard bounds, no weight load
python3 pipeline/cli/ms.py cases               # validate case data without a server
python3 pipeline/cli/ms.py doctor       # imports, size, collisions, mlx_lm, shadowing, litter, docs
python3 pipeline/cli/ms.py eval --out-dir DIR --model LABEL   # all eval suites -> TOML reports
python3 pipeline/cli/ms.py gate --base DIR --cand DIR         # promotion gate (pipeline/eval/gate.toml)
```

Commands are hot-loaded from `pipeline/cli/commands/`; doctor checks from
`pipeline/cli/doctor/checks/`. Discovery is `pipeline/cli/plugins/discovery.py`
(`load_plugins(folder, attr, pattern="**/*.py")`): non-underscore `*.py` files at
any depth carrying a manifest dict (`{"name": str, "run": callable, "help"?}`)
are loaded fresh every call — repeated calls see creates/edits/removes; duplicate
names and malformed manifests raise. `pipeline/cli/_load_plugins.py` is a compat
re-export only. `moe_partition` modes use the same loader with pattern
`**/manifest.py` under `surgery/moe_partition/modes/`.

## Surgery (`pipeline/surgery/`)

| Script | What it does |
|---|---|
| `depth/prune.py` | Remove transformer blocks by index or ablation rank |
| `width_prune.py` | Slice dense FFN intermediate_size (bf16 and any bit-width) |
| `mlp/only_prune.py` | Zero MLP weights (incl. quant scales/biases) in selected layers |
| `expert/prune.py` | Drop weakest MoE experts (stacked `switch_mlp` or per-expert keys) |
| `moefy/entry.py` | Convert dense FFN layers to MoE |
| `moe_partition/run.py` | Expert partitions per layer; geometry via required flags `--experts --expert-width --shared-width --group-size`; modes from `modes/**/manifest.py` |
| `merge_ties.py` | TIES + DARE merge of two same-arch fine-tunes |
| `abliterate.py` | Project out refusal direction |
| `neuron/edit.py` | Patch neuron output columns via activation contrast |
| `neuron/recipe.py` | Steering recipe from target vs baseline prompts |

## Profile (`pipeline/profile/`)

| Script | What it does |
|---|---|
| `block/distance.py` | Angular distance between block input/output (cut candidates) |
| `block/ablation.py` | Per-block loss delta: spec / import / distance proxy / score |
| `expert/routing_profile.py` | Expert routing frequencies + merge candidates |
| `mlp_stats.py` | MLP statistics for MoE conversion |
| `verify/lineage.py` | Prove pruned checkpoint = parent minus removed layers |
| `verify/origin.py` | Sample common quantized matrices with shuffled controls; similarity, not provenance proof |
| `neuron/probe.py` · `neuron/heatmap.py` | Per-layer neuron activations + HTML heatmap |
| `depth/route.py` | Mixture-of-Depths per-layer importance |
| `fast/bench.py` | Decode tok/s probe against served model |

**Blocked until tensor-array grows routes** — `expert/routing_profile.py` is already
ported (HOPE stats via `TARR_EMBEDDINGS=1 TARR_HOPE_STATS`). The three remaining
legacy `mlx_lm` interfaces are now fail-closed `NotImplementedError` stubs:
`profile/_hidden.py` `capture()` (needs per-layer residual states → blocks
`block/distance.py`), `profile/layer/_tap.py` `capture_mlp()` (needs MLP
input/activation taps → blocks `mlp_stats.py`), `profile/ablation/_forward.py`
`forward_logits()` (needs teacher-forced logits with layer-skip → blocks
`block/ablation.py --score`, `layer/ablation_serve.py`). See
`pipeline/profile/_mlx_load.py` header. `neuron/probe.py` needs a `hidden_states`
route tensor-array does not have yet.

## Eval (`pipeline/eval/`)

| Suite | Cases | Metric |
|---|---|---|
| `tool/run.py` | 104 | tool selection accuracy + argument validity |
| `code/run.py` | 17 | Python / frontend / review graded by required tokens |
| `rev/run.py` | 16 | x86 / ARM64 / formats / exploitation / decompiler |
| `kl/run.py` | ~30 prompts | mean KL(parent ‖ student) on general text |

Cases live in `<suite>/cases/<category>.py` (one `CASES` list per file, hot-loaded).
Loaders reject duplicate cases/IDs, empty selections and impossible tool expectations.
Reports preserve numeric precision and include an evaluation-source fingerprint.
Current graders are diagnostic: token matching does not prove code execution or reversing,
and KL compares renormalized top-k heads. Reports declare `diagnostic_only = true`.
The promotion gate rejects these reports, legacy reports without evidence metadata,
changed fingerprints/counts, non-finite scores, and empty/report-only rules.
There is currently no promotion-grade eval path. See [CLI details](docs/cli.md).

## Heal (`pipeline/heal/`)

| Script | What it does |
|---|---|
| `gen/corpus.py` | Build distillation JSONL |
| `cache_teacher_logits.py` | Top-32 teacher logprob harvest via tensor-array |
| `tool_distill.py` | Tool-routing LoRA via tensor-array-train (CE + top-32 KL) |
| `general_distill.py` | General corpus pass, lower LR, after tool_distill |
| `fuse_adapter.py` | Fuse LoRA adapter via `tensor-array-train fuse` |
| `run_heal.sh` | Corpus → cache → tool_distill → fuse |

## Export (`pipeline/export/`)

| Script | What it does |
|---|---|
| `mlx/convert.py` · `mlx/convert_stream.py` | Mixed-bit MLX quantization (recipes) |
| `repack/1bit_to_2bit.py` | Lossless 1-bit → 2-bit repack for Metal kernels |
| `bench/entry.py` | Rough speed + tool accuracy delta vs baseline |

## Environment variables

| Variable | Purpose |
|---|---|
| `SURGERY_TEACHER_MODEL` | Teacher (parent) model dir |
| `SURGERY_PARENT_MODEL` | Parent model dir (tokenizer fallback) |
| `SURGERY_DRAFT_MODEL` | Small model for fast draft stage |
| `SURGERY_TEACHER_CACHE` | Teacher logit NPZ path |
| `SURGERY_CACHE_DIR` | Teacher logit cache dir |
| `TARR_EVAL_PORT` | tensor-array port for evals/benches (default 8765) |
| `MS_KL_CACHE` | KL parent-trace cache path |
| `TARR_XMEM_DIR` | tensor-array external memory dir |

## Requirements

Python 3.11+ (tomllib), MLX (`mlx.core`), tensor-array server + `tensor-array-train`,
transformers (tokenizer only). No `mlx_lm` / `mlx_vlm`.

## License

MIT

# Profile (`pipeline/profile/`)

| Script | What it does |
|---|---|
| `block/distance.py` | Angular distance between block input/output (cut candidates) |
| `block/ablation.py` | Per-block loss delta: spec / import / distance proxy / score |
| `expert/routing_profile.py` | Expert routing frequencies + merge candidates |
| `mlp/stats.py` | MLP statistics for MoE conversion |
| `verify/lineage.py` | Prove pruned checkpoint = parent minus removed layers |
| `verify/origin.py` | Sample common quantized matrices with shuffled controls; similarity, not provenance proof |
| `neuron/probe/core.py` · `neuron/heatmap.py` | Per-layer neuron activations + HTML heatmap |
| `depth/route.py` | Mixture-of-Depths per-layer importance |
| `fast/bench/core.py` | Decode tok/s probe against served model |

**Runtime probe routes (2026-11-07).** Start the server with `TARR_PROBES=1` (renamed
runtimes: their own probe gate). `block/distance.py` posts cuts to
`/v1/internals/layer_distance`; `block/ablation.py --score-api` posts to
`/v1/layer_ablation` (layer skip = identity residual, teacher-forced mean NLL).
Both run server-side; hidden states never cross HTTP. Archs that ignore probes answer 501.
Implemented for the Qwen3.5/3.6 MoE forward only. Live check: AIMER-64e, both tools rank
layers 24-27 safest (distance 0.119, NLL delta 0.077) — diagnostic, not promotion evidence.

**Still blocked:** `expert/routing_profile.py` already works (HOPE stats via
`TARR_EMBEDDINGS=1 TARR_HOPE_STATS`). In-process stubs stay fail-closed:
`profile/_hidden.py` `capture()`, `profile/ablation/_forward.py` `forward_logits()`.
`profile/layer/tap/_core.py` `capture_mlp()` remains a stub. MLP inputs for dense-to-MoE come from
`profile/moefy/capture.py` via `/v1/internals/linear_inputs` (Qwen3 dense/MoE archs): MLP input,
attention input and per-head outputs per layer — feeds dense-to-MoE, Wanda and head ranking.

**Neuron probe.** `neuron/probe/core.py` posts to `/v1/internals/hidden_states` (mean-pooled
residual entering each layer; key N = last layer output) and writes records that
`neuron/heatmap.py` renders directly. Values are residual dimensions (hidden_size), not
MLP intermediate neurons — do not feed their indices to down_proj `--columns`.
`expert/usage.py` is retired (it wrote fake uniform usage); use `expert/routing_profile.py`.

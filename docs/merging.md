# Merging (`pipeline/surgery/merge/ties/core.py`, `passthrough/`, `soup/`)

`python3 merge/ties/core.py BASE FT1 [FT2 ...] DEST --method M [--weights ...] [--density 0.5] [--lambda 1.0]`

| Method | Idea | Needs |
|---|---|---|
| `task_arithmetic` | base + λ·Σ wᵢ·(FTᵢ − base) | ≥1 fine-tune |
| `ties` | trim small deltas, elect sign per weight, average agreeing deltas | ≥1 |
| `dare` | drop deltas at random (1 − density), rescale survivors | ≥1 |
| `dare_ties` | DARE drop, then TIES sign election | ≥1 |
| `breadcrumbs[_ties]` | drop smallest AND largest `--gamma` outlier deltas | ≥1 |
| `della[_linear]` | magnitude-ranked drop probability (`--epsilon` spread) | ≥1 |
| `linear` | Model Soup: weighted average of fine-tunes | ≥1 |
| `slerp` | spherical interpolation, `--lambda` = t | exactly 2 models |
| `nuslerp` | row-wise SLERP of two task vectors, added to base | exactly 2 fine-tunes |
| `karcher` | Riemannian (Karcher) mean on sphere, mean norm | ≥2 fine-tunes |
| `model_stock` | angle between task vectors sets pull toward base | ≥2 fine-tunes |

Other assembly tools:

| Script | What it does |
|---|---|
| `passthrough/entry.py OUT --slices "0:0-15,1:8-23" M0 M1` | Stack layer ranges from compatible models (frankenmerge / depth up-scaling) |
| `soup/greedy/core.py` | Greedy soup: add candidates in score order while held-out score improves |

All math runs on CPU (`mx.set_default_device(mx.cpu)`): shared GPU hit watchdog on big
dequantize/save. Quantized sources are dequantized, merged in float32, requantized unless
`--bf16-only`.

## Results

Two-domain merge lab (2026-11-08), Qwen3-0.6B. Fine-tune A = code/man continuation,
fine-tune B = reasoning chat (LoRA r16, 64 steps each, fused). Score A = held-out NLL on
code/man; score B = loss on held-out reasoning chat. Lower is better. All merges `--bf16-only`.

| Model | A | B |
|---|---|---|
| base | 2.4346 | 5.3769 |
| fine-tune A | 2.1586 | 2.2266 |
| fine-tune B | 2.5794 | 1.2151 |
| `nuslerp` | 2.2334 | 1.1650 |
| `ties` | 2.2483 | 1.1751 |
| `della` | 2.2689 | 1.1758 |
| `dare_ties` | 2.2753 | 1.1764 |
| `breadcrumbs` | 2.2248 | 1.1772 |
| `task_arithmetic` | 2.2978 | 1.2171 |
| `dare` | 2.3176 | 1.2417 |
| `della_linear` | 2.3151 | 1.2417 |
| `breadcrumbs_ties` | 2.2253 | 1.3135 |
| `karcher` | 2.2319 | 1.3401 |
| `linear` | 2.2319 | 1.3406 |
| `slerp` (t 0.5) | 2.2317 | 1.3407 |
| `model_stock` | 2.4343 | 5.3729 |

Every method except Model Stock beats base on both sets; sign-electing methods
(`ties`, `dare_ties`, `della`, `nuslerp`) beat fine-tune B on its own set while keeping
most of A. Model Stock returns base here by its own formula: two independent LoRA task
vectors are orthogonal (layer 10 q_proj cosine −0.0009, so pull t ≈ 0).

Search on mean of A and B (8 evals each):

| Search | Best mean | Result |
|---|---|---|
| greedy soup | 1.7862 | kept both = `linear` |
| `evolve/entry.py --method ties` (1+1)-ES | 1.7117 | start genome (density 0.5, λ 1); 7 mutations 1.7120–1.7382 |

At this budget search did not beat default TIES.

MoE combine (`moe/combine/entry.py`, one expert per fine-tune, router rows from each domain's
texts captured on base, attention from q8 base):

| Model | A | B |
|---|---|---|
| q8 base | 2.4401 | 5.3170 |
| combine top-1 | 2.2227 | 1.9503 |
| combine top-2 | 2.2276 | 1.9989 |

Routing is right (tokens sent to own expert: 81–96 % A, 96–98 % B across layers, in-sample),
yet B trails fine-tune B: combine keeps base attention, so fine-tunes' attention deltas are lost.
Test: same top-1 combine, attention from 8-bit `linear` merge of both fine-tunes as `base`:

| Model | A | B |
|---|---|---|
| 8-bit `linear` merge | 2.2354 | 1.3419 |
| combine top-1, base attention | 2.2227 | 1.9503 |
| combine top-1, merged attention | 2.1881 | 1.3180 |

Merged attention closes most of B gap and beats merge alone on both sets. Pass a merge of
the fine-tunes as `base` whenever fine-tunes trained attention.

Earlier 4B check (Qwen3.5 4B, pruned + healed pair, old tokenizer): SLERP t=0.5 and
task arithmetic midpoint both 2.171 held-out loss.

# Performance (2026-11-08)

Apple M4, 32 GB. One model at a time, nothing else running. Per model: warmup, then 3 greedy
128-token chat replies; tok/s = completion tokens ÷ request wall time (includes ~20-token
prefill). Median of 3 shown; spread in `bench/perf.txt`. N-gram drafting off: damaged models
repeat text, drafts are then always accepted and speed looks 5× higher than model compute.
Quality numbers live in each method's doc; this page is speed and memory only.

### Qwen3-0.6B

| Model | tok/s | Peak RAM | Size |
|---|---|---|---|
| bf16 | 60.6 | 1710 MB | 1.4 GB |
| 8-bit | 48.2 | 1597 MB | 773 MB |
| 4-bit RTN / AWQ / GPTQ | 51.3 / 51.7 / 50.4 | 1392 MB | 567 MB |
| heads −2 / −4 of 8 KV groups | 66.2 / 72.7 | 1601 / 1491 MB | 1.3 GB |
| hidden 1024 → 896 | 69.3 | 1527 MB | 1.2 GB |
| ASVD 0.9, dense reconstruction | 61.3 | 1708 MB | 1.8 GB |
| ASVD 0.9, factored | 64.7 | 1622 MB | 1.3 GB |
| MoE 8 experts, top-8 / 4 / 2 | 40.9 / 43.3 / 44.1 | 1277 MB | 773 MB |
| MoE top-4 healed (expert LoRA fused) | 36.1 | 1358 MB | 783 MB |
| combine top-1 / top-2 | 41.3 / 37.0 | 1540 MB | 1.0 GB |

At 0.6B decode is not weight-bandwidth bound: 4-bit is no faster than bf16, and MoE routing
overhead outweighs skipped experts. Smaller structure (heads, hidden) helps directly.

### Qwen3.5 4B

| Model | tok/s | Peak RAM | Size |
|---|---|---|---|
| bf16 | 9.5 | 8308 MB | 7.9 GB |
| 4-bit | 30.6 | 2592 MB | 2.2 GB |
| linear-attention heads −4 / −8 (bf16) | 10.4 / 11.3 | 7799 / 7287 MB | 7.4 / 6.9 GB |
| full-attention −1 KV group (bf16) | 9.6 | 8165 MB | 7.7 GB |
| MoE 8 experts, top-8 / 4 / 2 (4-bit) | 30.7 / 38.0 / 37.3 | 2592 MB | 2.2 GB |

At 4B, top-4 MoE is 24 % faster than dense 4-bit (quality cost in [moefy](moefy.md)).

### Llama 3.2 3B

| Model | tok/s | Peak RAM | Size |
|---|---|---|---|
| 4-bit | 21.4 | 3492 MB | 1.7 GB |
| Mixtral layout, top-8 / top-4 | 13.5 / 14.3 | 2888 MB | 1.7 GB |

Runtime Mixtral-layout expert path is slower than dense here; MoE gives no speed win on Llama yet.

### Sparse 2:4 kernel (`sparse24-bench`, 3 runs)

| Matrix N×K | Dense bf16 | 2:4 | 4-bit dense |
|---|---|---|---|
| 3072×1024 | 1.52–1.73 ms | 1.02–1.32 ms | 0.63–0.73 ms |
| 9216×2560 | 8.85–10.99 ms | 5.73–5.77 ms | 2.76–2.87 ms |
| 2560×9216 | 8.94–9.90 ms | 5.46–6.05 ms | 2.70–3.15 ms |

2:4 is 1.5–1.7× faster than dense bf16 with exact output, but plain 4-bit is ~2× faster than 2:4.
Bench-only kernel (runtime src has no 2:4 path): 4-bit already wins at lower quality cost (see [surgery](surgery.md)).

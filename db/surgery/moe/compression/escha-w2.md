<!-- escha-w2.md — Escha-W2: 2-bit Qwen3.6-35B-A3B with fine-tune recovery (same base as ours). -->
<!-- NEVER DELETE COMMENTS — explain every non-obvious line. Update comments with the code; do not strip them. -->

# Escha-W2 — 2-bit quant + recovery on our exact base

Source: @Eschalabs, X, 2026-07-28 — https://x.com/Eschalabs/status/2082223824284856324

Model: huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2 (Apache-2.0)
Runtime: huggingface.co/EschaLabs/escha-runtime-qwen3moe

## Claim (vendor-reported, not verified)

| Item | Value |
|---|---|
| Base | Qwen3.6-35B-A3B (all 256 experts kept) |
| Size | 12.3 GB on disk |
| Quality | ~100% of FP8 avg over 12 benchmarks |
| BFCL tool use | 88.9 |
| Speed | 225 tok/s single stream, RTX 4090 |
| Build time | ~10 h base → final checkpoint |

Method, in their words: "low-bit quantization with model-aware fine tuning and
recovery to preserve capabilities most vulnerable to low-bit error". No paper.

Format details: [format and runtime notes](format.md).

## Their M4 base 24 GB head-to-head (bench/results/m4-base-24gb/head_to_head.json)

| Model | Resident | Decode @512 | Prefill @2048 |
|---|---|---|---|
| Escha W2 | 12.25 GB | 26.4 tok/s | 236 tok/s |
| stock MLX 4-bit | 19.51 GB | 42.7 tok/s | 339 tok/s |

Smaller but ~38% slower decode on Mac: codebook decode costs more than plain 4-bit.

## Relevance for us

Same base, same goal (shrink without losing tool use), and it keeps every expert.
Supports lesson from SLAB win: cut BITS + heal, not expert COUNT.
Recipe shape = quantize → recovery fine-tune → eval tool calls. Matches our
heal order (`../healing/INDEX.toml`). 12.3 GB vs SLAB 21 GB.

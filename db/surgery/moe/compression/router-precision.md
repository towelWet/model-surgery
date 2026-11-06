<!-- router-precision.md — keep MoE router (gate) weights at high precision when quantizing. -->
<!-- NEVER DELETE COMMENTS — explain every non-obvious line. Update comments with the code; do not strip them. -->

# Router precision — do not quantize the gate

Source: @bnjmn_marie, X, 2026-10-14 — https://x.com/bnjmn_marie/status/2110440416336310276

## Claim

On LFM2.5-24B-A2B, quantizing MoE routers 16-bit → 8-bit caused the largest damage
he had seen: accuracy dropped and next-token distribution shifted a lot. That shift
also hurts speculative-decoding acceptance for drafters trained against BF16.

Routers are ~10 MB total in BF16; 8-bit saves ~5 MB. Not worth it.

## Measured on our builds (2026-11-02)

| Build | Experts / rest | Routers (80 entries) |
|---|---|---|
| Qwen3.6-35B-A3B-SLAB (intact) | 4-bit g64 | 8-bit g64 |
| Tau3.6-35B-A3B-hope224r20-SLAB | 4-bit g64 | 8-bit g64 |

Read from each `config.json` `quantization` map (`mlp.gate` + `shared_expert_gate`).

## Caveat

Damage is measured on LFM2.5, not Qwen3.6. Untested on our model.
Both builds share 8-bit routers, so this cannot explain intact vs pruned gaps;
it may lower both.

## Use

Rebuild with routers at bf16 when re-quantizing. Pairs with `db/surgery/moe/healing/router-kd.md`
(router-only training wants a full-precision router to train).

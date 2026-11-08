# GGUF editing (`pipeline/gguf/`)

Byte-level GGUF v2/v3 reader and v3 writer. No dequantization: every kept tensor and
every kept expert slice is copied byte-for-byte, so all ggml quant types work.

| Command | What it does |
|---|---|
| `ms gguf inspect FILE [--tensors]` | Header, scalar hyper-parameters, tensor table |
| `ms gguf prune-layers SRC DEST --layers N ...` | Drop blocks, renumber survivors, shrink per-layer arrays |
| `ms gguf prune-experts SRC DEST --drop E ...` | Drop routed experts from every layer |
| `ms gguf prune-experts SRC DEST --plan plan.json` | `{"layer": [kept experts]}`; equal count per layer |

Expert pruning slices tensors containing `_exps.`, `ffn_gate_inp.` or `exp_probs_b.`
along their last (expert) dimension; shared experts (`shexp`) are untouched. It refuses
keeping fewer experts than `expert_used_count`. Layer pruning refuses cutting
`nextn_predict_layers` (MTP) blocks. Destinations are never overwritten.

Live check (2026-11-07, llama.cpp built from local source):

| Edit | Model | Loads | tok/s | PPL (llama.cpp docs, 16×512) |
|---|---|---|---|---|
| none (read+write) | Qwen3 4B Q4_K_XL | byte-identical | — | — |
| cut layers 24–27 | Qwen3 4B Q4_K_XL | yes | 23.4 → 28.9 | 5.08 → 8.93 |
| drop experts 24–31 of 32 | Granite 3.1 1B-A400M Q4_K_M | yes | 73.7 → 161.5 | 6.15 → 10.78 |

Cuts above were arbitrary (no profiling); quality loss shows why profile → cut → heal matters.
## Dequantize → edit → requantize

| Command | What it does |
|---|---|
| `ms gguf to-safetensors SRC DEST --hf-ref HF_DIR` | gguf-py dequantize every tensor → HF bf16 safetensors; names chosen from reference's own tensor names; config patched from GGUF metadata |
| `ms gguf from-safetensors HF_DIR OUT.gguf --type Q4_K_M` | llama.cpp HF-to-GGUF converter (f16) then `llama-quantize` |

Between them every safetensors tool applies (width/heads/hidden/neuron edits, abliteration,
merging). Paths: `[runtime.llamacpp] source/bin` in the private toml (`private-toml path model-surgery`) or `LLAMA_CPP_SOURCE`/`LLAMA_CPP_BIN`.
Llama: converter permutes q/k rows for rope; dequantize un-permutes them back to HF order
(`pipeline/gguf/hf/unpermute_qk.py`; unit test inverts llama.cpp's permutation exactly).
Real round trip (2026-11-08): Llama 3.2 3B bf16 → F16 GGUF → safetensors: 254/254 names,
max relative error 0.

Round trip check (2026-11-07): Qwen3-0.6B bf16 → F16 GGUF → safetensors: same 311 names,
max relative error 2e-7.

Quality through round trip and edit (2026-11-08, Qwen3-0.6B, `llama-perplexity -c 512` held-out):

| GGUF | PPL |
|---|---|
| F16 | 11.0624 |
| Q4_K_M | 11.9336 |
| Q4_K_M → safetensors → Q4_K_M | 11.9217 |
| FFN width 3072 → 2560, then Q4_K_M (no healing) | 17.4830 |

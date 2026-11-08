# Pruning and compression (`pipeline/surgery/`)

| Script | Granularity | Method |
|---|---|---|
| `depth/prune.py` | whole blocks | ablation rank or index list |
| `width/prune.py` | FFN neurons | slice `intermediate_size` |
| `expert/prune/core.py` | MoE experts | drop weakest routed experts |
| `heads/entry.py --drop N --calib DIR` | KV groups | drop N query-head groups + their K/V head, scored by captured attention outputs |
| `heads/linear/entry.py --drop N --calib DIR` | linear-attention key groups | Qwen3.5/3.6 Gated DeltaNet: drop N key heads + value heads they feed (q/k/v rows, conv1d channels, z, a/b, A_log, dt_bias, out_proj columns); full-attention layers untouched |
| `hidden/entry.py --keep H --calib DIR` | residual channels | Minitron-style: keep H channels with largest summed normed activations |
| `sparse/entry.py --sparsity S [--nm 2:4] [--roles ...] [--calib DIR]` | single weights | magnitude, or Wanda (`|W|` × input-channel norm, compared per row); N:M pattern |
| `sparse/remask/core.py PRUNED HEALED DEST` | single weights | re-zero pruned positions after heal + fuse |
| `lowrank/entry.py --ratio R [--roles ...] [--calib DIR --alpha 0.5]` | matrix rank | SVD, or activation-aware ASVD; writes dense reconstruction + `lowrank/factors.safetensors` |
| `vocab/entry.py --corpus FILE...` | vocabulary rows | keep tokens corpus needs + byte alphabet + added tokens + BPE merge parents |

Calibration `DIR` comes from `pipeline/profile/moefy/capture.py` (`attn.<i>`, `o.<i>`, `mlp.<i>` taps).
Runtime taps exist for Qwen3 dense, Qwen3.5/3.6 hybrid (dense + MoE; `o` = out_proj input in
linear-attention layers, gated head output in full-attention layers) and Llama 3.x. Checked
2026-11-07 by real capture: 28 layers x 3 taps on Qwen3.5 4B and Llama 3.2 3B.

All edits refuse quantized sources (prune bf16, then quantize) except vocab pruning, which
slices token rows of packed weights directly. `dequant/entry.py SRC DEST` turns MLX affine
checkpoint into bf16 start point (quantization rounding stays baked in: compare edits against
this dequantized dense, not fp original). Destinations are never overwritten.

## Limits

Zeros (sparse) do not shrink safetensors or speed dense kernels. Low-rank dense
reconstruction saves nothing until a runtime serves factors directly. Head pruning keeps
the same group count in every layer (config has one `num_key_value_heads`).
Vocabulary pruning keeps text outside corpus encodable (byte alphabet) but splits it into
more tokens, so quality on that text drops.

## Results — Qwen3-0.6B, 32 held-out pieces (code + man pages), runtime NLL

Measured 2026-11-07 after runtime tokenizer fix (see below). Dense bf16 = **2.4346**.

| Model | Change | NLL |
|---|---|---|
| q8 | 8-bit affine | 2.4401 |
| Wanda 50 % all linears → LoRA KD heal → fuse | 0 % zero (fuse fills zeros) | 2.4926 |
| same, then `sparse/remask/core.py` | 50 % zero restored | 2.5398 |
| Wanda 50 % MLP only | 50 % MLP weights zero | 2.7518 |
| Wanda 50 % all linears | 50 % weights zero | 2.8469 |
| hidden 1024 → 896 | −12.5 % residual width | 3.1204 |
| Wanda 2:4 all linears | 2 of every 4 zero | 4.8007 |
| heads −2 of 8 KV groups | −25 % attention heads | 5.1321 |
| magnitude 50 % MLP | 50 % MLP weights zero | 6.7940 |
| heads −4 of 8 KV groups | −50 % attention heads | 8.0808 |
| ASVD MLP, 50 % params | rank 384 of 1024 | 10.3232 |
| SVD MLP, 50 % params | rank 384 of 1024 | 14.1877 |

Only first two rows were healed (64 steps, rank 16, KD from dense, teacher cache rebuilt after
tokenizer fix). Zero fractions checked on saved weights: 0.50 → 0.00 after fuse → 0.50 after remask.
ASVD beats plain SVD at same rank (activation-weighted truncation), but 50 % rank cut is
far too aggressive without healing.

ASVD MLP by parameter ratio (2026-11-08, no healing; dense 2.4346):

| `--ratio` | Rank of 1024 | Rel. error | NLL |
|---|---|---|---|
| 0.9 | 640 | 0.357 | 6.3151 |
| 0.8 | 576 | 0.410 | 7.3770 |
| 0.5 | 384 | 0.577 | 10.3232 |

Rank = ratio·out·in/(out+in), rounded down to 64 for quant groups (0.9 → 691 → 640).
Negative result: MLP spectra are flat, so even ratio 0.9 cuts a third of rank. Low-rank
needs healing before use. Factored form serves same quality (0.9: 6.3040).

### Qwen3.5 4B (hybrid Gated DeltaNet), held-out NLL, no healing

| Model | NLL |
|---|---|
| bf16 | 1.8099 |
| 4-bit | 1.8100 |
| full-attention −1 of 4 KV groups (`heads/`) | 1.9392 |
| linear-attention −4 of 16 key heads (`heads_linear/`) | 2.0053 |
| linear-attention −8 of 16 key heads | 2.6485 |

MoE conversions of same model: [moefy](moefy.md).

### Vocabulary pruning — loss per character (tokenizer-independent)

`pipeline/profile/vocab/nll_chars.py`: total nats over identical held-out text ÷ characters.

| Model | Vocab | Size | Held-out tokens | nats/char |
|---|---|---|---|---|
| dense | 151,669 | 1.50 GB | 8,991 | 0.5654 |
| corpus = 64 training pieces | 5,028 | 0.90 GB | 10,737 | 0.8945 |
| corpus = large stdlib/man set | 13,564 | 0.94 GB | 8,991 | 0.5564 |

Large corpus covered every held-out token, so text tokenizes identically and loss drops
slightly (softmax over fewer rows). Small corpus misses tokens held-out text needs.

## Runtime tokenizer fix (2026-11-07)

Vocabulary pruning exposed runtime pre-tokenizer bug: Split scanner did not follow
Qwen/GPT-4 regex for whitespace (`" =="` split into `" "` + `"=="`; indentation runs
kept whole; `"\nfoo"` one piece). Fixed in both runtimes with `tests/test-pretok-split.cpp`
(expected pieces generated by HF `tokenizers`). Effect on dense Qwen3-0.6B held-out:
0.7687 → 0.5654 nats/char (−26 %). All NLL numbers before this date used old tokenizer;
tables above are re-measured.

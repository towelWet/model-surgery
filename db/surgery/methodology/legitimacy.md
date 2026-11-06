# Legitimacy & architecture traps

## Checklist

Checked items below are historical records with dates where recorded; they are not current verification — re-run before citing.

- [ ] Base model confirmed in HuggingFace model tree
- [ ] FFN arithmetic: `E × S + H == parent_W` (partition) or shard count verified
- [ ] Active param ratio stated and plausible
- [ ] Only router/gate trained (not full fine-tune disguised as conversion)
- [ ] Honest eval table with regressions
- [ ] tensor-array arch class registered before pulling weights
- [x] Packed GEMM correctness: bits=1 K=4096/5120/256 pass, ternary K=4096/5120/256 pass
- [x] Metal shader syntax: `xcrun metal -c` exit 0 for 1-bit + ternary
- [x] God `/health`+`/healthz`: built and cargo-checked
- [x] Crackme suite: 6/6 (retroactive quote-strip)
- [x] ask.py parser: 6 unit tests pass (quotes, backtick, prefix, first-word).
- [x] parser unit test: 6/6 (test_parser.py, no POST).
- [x] `perf-monitor.sh`: standalone, `bash -n` clean
- [x] `pipeline/profile/fast/bench.py`: TTFT separated, streaming, exit 0/1
- [x] `preflight.sh`: 11/11 pass (11 checks; overnight 07:50)
- [x] OVERNIGHT_GUIDE.md [missing: not on disk 2026-11-06] exists (OVERNIGHT_GUIDE.md [missing: not on disk 2026-11-06])

## `model_type` naming traps

Reflects MoE **infrastructure**, not base family.

| model_type | Meaning |
|---|---|
| `qwen3_5` | Qwen3.8-27B dense (Qwen3.5 arch internals) |
| `qwen3_5_moe` | Qwen3.5-MoE routing on some base |
| `qwen3_moe` | Qwen3-30B-A3B / 235B-A22B family |

Verify via model tree, not `model_type` string.

## tensor-array arch support

```bash
rg "Qwen3_5Moe\|Qwen3Moe\|MoeFor" <tensor-array-src>/arch/ --type cpp
```

Register target arch before pulling weights. Tau serve arch:
`TARR_ARCH=Qwen3_5ForConditionalGeneration` — see OPERATOR.md [missing: not on disk 2026-11-06].

## Packed-GEMM (host, 2026-09-21)

- [x] bits=1: K=5120 **pass**; edge K=4096/256 **pass**
- [x] ternary bits=2: K=5120 **pass**; edge K=4096/256 **pass**
- [x] Metal shader syntax: `xcrun metal -c` exit 0 (1-bit + ternary)
- [x] Bonsai dispatch shape check: PASS (gate_proj verified, all linears multiples of 32 and 128).
- [ ] Metal JIT runtime — not run (morning)

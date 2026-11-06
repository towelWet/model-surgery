# Papers, related projects, reference trees

## Key papers

- [ToMoE: Dense→MoE via dynamic structural pruning](https://arxiv.org/abs/2501.15316)
- [ExpertWeaver: MoE from GLU activation patterns](https://arxiv.org/pdf/2602.15521)
- [DOT-MoE: Optimal transport expert assignment](https://arxiv.org/pdf/2606.01666)
- [Drop-Upcycling: Sparse MoE with partial re-init](https://arxiv.org/pdf/2502.19261)
- [ConMoE: Expert pool consolidation](https://arxiv.org/pdf/2605.29350)
- [Prune+Distill MoE→Dense](https://arxiv.org/html/2605.28207v1)
- [Awesome-Efficient-MoE list](https://github.com/pprp/Awesome-Efficient-MoE)
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) — BitLinear, 1-bit weights.
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — ternary weights {-1,0,+1} (~1.58 bits).

## Related

- `/posthoc-moe` — retired; superseded by this skill
- tensor-array MoE: `src/arch/qwen/3_5/moe.cpp` (in tensor-array repo)
- Whittle-MoE-27B: `logic65/Qwen3.8-Whittle-MoE-27B-A17.8B`
- Whittle-Next-27B-A3B: Flash Next hashed n-gram + ~3B active (research)

## Reference codebases

`./Reference/` — phase notes + upstream clones.
Catalog: optional local Reference catalog (14 repos by phase + `whittle-logic65/`).
All torch/CUDA: port method, not code.

## Overnight 2026-09-21 (1-bit / ternary / eval)

- [mlx#3161](https://github.com/ml-explore/mlx/pull/3161) — affine 1-bit pack (`w_hat = scale * bit + bias`, LSB-first uint32). tensor-array: `include/tensor-array/affine-1bit.h`.
- PrismML 1-bit Qwen: `${MODEL_ROOT}/checkpoint` (Qwen3.6-27B). Ternary: [prism-ml/Ternary-Bonsai-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit). [prismml.com](https://prismml.com); whitepaper [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo).
- cpu-qmm host GEMM: tensor-array `src/quant/cpu-qmm.cpp` + `cpu-qmm-ternary.cpp`; `test-quant-affine-1bit` / `test-quant-packed-ternary` K=5120 **pass**.
- tensor-array cpu-qmm: custom 1-bit/ternary packed host GEMM, metal_kernel JIT dispatch (this codebase).
- [T-MAC](https://arxiv.org/abs/2407.00088) — LUT mpGEMM for low-bit / ternary (CPU). Host impl here is `cpu-qmm-ternary.cpp`, not T-MAC.
- mlx [`async_eval`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html): overnight `eval-boundary.cpp` `mx::eval`→`async_eval`; per-layer Qwen3.5 `forward_decoder.cpp`; submit-first `sampled.cpp`.

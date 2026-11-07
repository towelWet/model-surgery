# Escha-W2 format and runtime notes

## Format (read from HF config.json + index, 2026-11-02)

| Part | Storage |
|---|---|
| Routed experts gate_up | 2-bit codebook (`escha_code`) + `escha_rin`/`escha_rout` rotations + `escha_s_in`/`s_out` scales |
| Routed experts down | 3-bit, same scheme |
| Attention, shared expert | int8 + per-channel `weight_scale` |
| Router `mlp.gate` | plain `weight` (unquantized) |

## Runtimes

CORRECTED 2026-11-02: earlier text said "no MLX". Wrong. CUDA side = SGLang wheel + ZML
binary; Mac side = github.com/EschaLabs/escha-mlx (Apache-2.0, Python on MLX, own Metal
shaders in `github.com/EschaLabs/escha-mlx/escha_mlx/msl.py`, reference decoder `github.com/EschaLabs/escha-mlx/ref.py`). Serves same files, no conversion.
tensor-array cannot load it as-is: needs a port of that decoder + kernel.


# Environment variables

| Variable | Purpose |
|---|---|
| `SURGERY_TEACHER_MODEL` | Teacher (parent) model dir |
| `SURGERY_PARENT_MODEL` | Parent model dir (tokenizer fallback) |
| `SURGERY_DRAFT_MODEL` | Small model for fast draft stage |
| `SURGERY_TEACHER_CACHE` | Teacher logit NPZ path |
| `SURGERY_CACHE_DIR` | Teacher logit cache dir |
| `TARR_EVAL_PORT` | tensor-array port for evals/benches (default 8765) |
| `MS_KL_CACHE` | KL parent-trace cache path |
| `TARR_XMEM_DIR` | tensor-array external memory dir |

## Requirements

Python 3.11+ (tomllib), MLX (`mlx.core`), tensor-array server + `tensor-array-train`,
transformers (tokenizer only). No `mlx_lm` / `mlx_vlm`.

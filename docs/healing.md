# Heal (`pipeline/heal/`)

| Script | What it does |
|---|---|
| `gen/corpus.py` | Build distillation JSONL |
| `cache_teacher_logits.py` | Top-32 teacher logprob harvest via tensor-array |
| `tool_distill.py` | Tool-routing LoRA via tensor-array-train (CE + top-32 KL) |
| `general_distill.py` | General corpus pass, lower LR, after tool_distill |
| `fuse_adapter.py` | Fuse LoRA adapter via `tensor-array-train fuse` |
| `run_heal.sh` | Corpus → cache → tool_distill → fuse |

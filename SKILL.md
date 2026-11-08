---
name: model-surgery
description: Inspect and restructure safetensors checkpoints with modular CLI operations, quantization-aware edits, pruning, storage, and diagnostic eval. Runtime commands and private aliases are configured locally. Use for model-surgery and checkpoint surgery work.
---
# Model Surgery

The skill is the project in this directory. Read README.md first, then current.toml
and the relevant documentation. Preserve source checkpoints and existing local state.

## Required boundaries

- Never import or execute `mlx_lm` or `mlx_vlm`.
- Runtime code changes go into private source runtime only; public runtime is generated
  by release generator command. Never hand-edit or sed-mirror generated runtime tree.
- Local inference belongs to the configured runtime; product names imply no capabilities.
- Runtime paths, environment aliases and publication exclusions stay in `$CLAUDE_DIR/private/model-surgery.toml` (private-toml system; shape: `local.example.toml`).
- Follow profile → surgery → heal → export; unavailable profiling is a blocker.
- Dense-to-MoE: Qwen3 dense only; sparse top-K needs healing before any quality claim.
- Structured tool-call training needs an explicit capable runtime adapter.
- Eval reports are diagnostic-only; never promote a checkpoint on those scores.
- Tests, builds and byte preservation are not real-model functional proof.
- One function per code file; 50-line cap; run doctor after changes.
- Imports are absolute (`from pipeline.x import y`); scripts put only the repo root on sys.path.
  Repeated name words are folders (`wquant/awq/_fold.py`). Doctor `bare` enforces imports.
- No publication without target-specific authorization and a passing privacy gate.
- CrackMe experiments and their targets must be MoE, never dense.

## Commands

```sh
python3 pipeline/cli/ms.py list
python3 pipeline/cli/ms.py doctor
python3 pipeline/cli/ms.py runtime
python3 pipeline/cli/ms.py privacy
python3 pipeline/cli/ms.py inspect MODEL
python3 tests/run.py
```

## Documentation

- [Workspace and version lanes](docs/workspace.md)
- [Runtime configuration](docs/runtime.md), [CLI](docs/cli.md)
- [Surgery](docs/surgery.md), [pruning](docs/pruning.md), [merging](docs/merging.md), [GGUF](docs/gguf.md), [profile](docs/profile.md), [storage](docs/storage/overview.md)
- [Healing](docs/healing.md), [dense-to-MoE](docs/moefy.md), [expert split](docs/storage/experts.md), [evaluation](docs/evaluation.md), [export](docs/export.md)
- [Performance](docs/performance.md)
- [Audit and acceptance limits](docs/audit.md)
- [Methodology index](db/surgery/methodology/INDEX.toml)

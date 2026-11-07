---
name: model-surgery
description: Inspect and restructure safetensors checkpoints with modular CLI operations, quantization-aware edits, pruning, storage, and diagnostic eval. Runtime commands and private aliases are configured locally. Use for model-surgery and checkpoint surgery work.
---
# Model Surgery

The skill is the project in this directory. Read README.md first, then current.toml
and the relevant documentation. Preserve source checkpoints and existing local state.

## Required boundaries

- Never import or execute `mlx_lm` or `mlx_vlm`.
- Local inference belongs to the configured runtime; product names imply no capabilities.
- Runtime paths, environment aliases and publication exclusions stay in private local.toml.
- Follow profile → surgery → heal → export; unavailable profiling is a blocker.
- Dense-to-MoE conversion is disabled until trained routing and loader support exist.
- Structured tool-call training needs an explicit capable runtime adapter.
- Eval reports are diagnostic-only; never promote a checkpoint on those scores.
- Tests, builds and byte preservation are not real-model functional proof.
- One function per code file; 50-line cap; run doctor after changes.
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
- [Surgery](docs/surgery.md), [profile](docs/profile.md), [storage](docs/storage/overview.md)
- [Healing](docs/healing.md), [evaluation](docs/evaluation.md), [export](docs/export.md)
- [Audit and acceptance limits](docs/audit.md)
- [Methodology index](db/surgery/methodology/INDEX.toml)

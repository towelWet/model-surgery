# Model Surgery

Inspect and transform trained safetensors checkpoints through a modular CLI.
Weight operations use MLX; local runtime integration uses configured commands and
an OpenAI-compatible HTTP protocol. Backend names do not determine capabilities.

![CLI inspection checks](docs/figs/cli-checks.png)

## Run locally

```sh
python3 pipeline/cli/ms.py list
python3 pipeline/cli/ms.py doctor
python3 pipeline/cli/ms.py runtime
python3 pipeline/cli/ms.py privacy
python3 tests/run.py
```

## Documentation

- [CLI and evidence](docs/cli.md)
- [Surgery operations](docs/surgery.md)
- [Profiling and unavailable routes](docs/profile.md)
- [Modular checkpoint storage](docs/storage/overview.md)
- [Runtime adapters and private configuration](docs/runtime.md)
- [Healing](docs/healing.md), [evaluation](docs/evaluation.md), [export](docs/export.md)
- [Environment and dependencies](docs/environment.md), [workspace layout](docs/workspace.md)
- [Audit results and limitations](docs/audit.md)

## Acceptance limits

Follow profile → surgery → heal → export. Required per-layer instrumentation is
unavailable; dependent campaigns stay blocked. Tests on small synthetic checkpoints
establish transformation behavior, not real-model quality or promotion readiness.
Dense-to-MoE conversion requires trained routers and verified loader support and is
currently disabled. Structured tool-call training requires a capable runtime adapter.
DARE task arithmetic is implemented; TIES sign election is not. Direct GGUF and
PyTorch checkpoint editing, dynamic expert loading, and promotion-grade eval are absent.

Python 3.11+ is required. No `mlx_lm` or `mlx_vlm`. Double-click `clean.command`
to remove generated Python caches. Existing checkpoints are never edited in place.

## License

MIT

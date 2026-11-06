# CLI inspection and evidence

Run these from the skill directory; Python 3.11+ is required.

```sh
python3 pipeline/cli/ms.py inspect /path/to/checkpoint
python3 pipeline/cli/ms.py inspect /path/to/checkpoint --out summary.toml
python3 pipeline/cli/ms.py cases
python3 pipeline/cli/ms.py cases tool
python3 pipeline/cli/ms.py list surgery --search prune
python3 tests/run.py
```

`inspect` reads config and safetensors headers only. It supports single-file and
indexed checkpoints, detects duplicate tensors, invalid byte ranges, missing shards,
and inconsistent weight indexes. It does not validate payload values or run inference.
The output reports configured/observed layer counts, expert count, dtypes and storage.
`--out` refuses an existing file. Incomplete or nonstandard checkpoint layouts fail explicitly.

`cases` finds case suites recursively through their existing suite entrypoints,
validates duplicates and offered tools, and prints counts plus the source fingerprint.
Case loading performs the same validation before inference. Unknown/empty categories fail.

Every current eval report declares diagnostic-only status. Required-token grading does
not prove executable code or successful reversing; top-k KL is not full-distribution KL.
The gate refuses diagnostic and legacy evidence even when accuracy is high.
Fingerprints cover eval/profile Python source and tool-schema JSON; changed source or
case counts makes reports incomparable. Model labels remain caller supplied.

Weight similarity samples common quantized matrices, rather than fixed model layers:

```sh
python3 pipeline/profile/verify/origin.py MODEL_A MODEL_B --limit 8 --rows 256
```

`--tensor PREFIX` selects explicit matrices; `--limit 0` scans all supported matrices.
Mixed quantization settings are resolved per tensor prefix. This reports sample cosine
and a shuffled control; it does not prove common provenance or equivalent behavior.

Depth pruning validates layer ranges and existing destinations before creating output,
including under `python -O`; configs without `layer_types` are supported.

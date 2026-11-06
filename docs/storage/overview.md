# Modular Safetensors Storage (schema_version 1)

Backend-independent raw-byte checkpoint partitioning. No mlx/numpy/torch needed —
storage moves bytes, never interprets weights.

## Implemented

- `pipeline/storage/plan.py` `plan_split(source)` — validate + plan, no writes.
- `split.py` `split_checkpoint(src, dest, dry_run, resume)` — shard headers, one
  safetensors file per tensor under `tensors/<sha256(name)>.safetensors`, raw
  headers under `headers/<sha256(path)>.bin`, ancillary bytes under `assets/`.
- `reassemble.py` `reassemble_checkpoint(store, dest, resume)` — byte-identical
  rebuild: stored raw headers + payloads in original offset order.
- `inspection.py` `inspect_store(store, verify)` — manifest inventory; `--verify`
  rehashes every file. `ms weights split|join|inspect|recover` CLI verbs.

## Guarantee class

`lossless-bytes` only: split+join restores identical shard/asset bytes proven by
sha256. This says nothing about inference equivalence, model quality, or memory
reduction. Shape equality never implies functional compatibility.
Markers (transaction/manifest/complete) are integrity checks, not security
boundaries. The store format does not establish runtime load support; loading
stays owned by consumer runtimes such as tensor-array.

## Transaction safety

transaction.toml first, manifest.toml after all hashes verify, complete.toml
LAST (manifest sha256). Interrupted stores stay incomplete; consumers refuse them.
`--resume` requires matching operation+fingerprint; damaged partials are never
overwritten. Destinations need `Tau` in the name; overlap with sources refused.

## Not implemented (proposal only)

Per-expert layouts, runtime lazy loading/residency (tensor-array owns), GGUF
adapters, MSIR/dependency graphs, compatibility engine, virtual checkpoints,
selective execution, training-state reconstruction. Proposed roadmap tracked in
`current.toml`; none of it exists here yet.

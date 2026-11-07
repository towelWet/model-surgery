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
- `inspection.py` `inspect_store(store, verify)` — verifies every recorded file by
  default. `--inventory-only` skips hashes and never reports completion.
  `ms weights split|join|inspect|recover` CLI verbs; incomplete recovery exits nonzero.

## Guarantee class

`lossless-bytes` only: split+join restores identical shard/asset bytes proven by
sha256. Every regular non-shard file is preserved recursively, including multimodal
assets, custom loader code, hidden files and unindexed auxiliary weights. Symlinks,
special files and reserved transaction-name collisions are refused before writes.
Empty directories and filesystem metadata are outside this byte contract.
This says nothing about inference equivalence, model quality, or memory
reduction. Shape equality never implies functional compatibility.
Markers (transaction/manifest/complete) are integrity checks, not security
boundaries. The store format does not establish runtime load support; loading
stays owned by consumer runtimes such as tensor-array.

## Transaction safety

transaction.toml first, manifest.toml after all hashes verify, complete.toml
LAST (manifest sha256). Interrupted stores stay incomplete; consumers refuse them.
`--resume` requires matching operation+fingerprint; damaged partials are never
overwritten. Destinations accept general names; overlap with sources is refused.

## Not implemented (proposal only)

Per-expert layouts, runtime lazy loading/residency (tensor-array owns), GGUF
adapters, MSIR/dependency graphs, compatibility engine, virtual checkpoints,
selective execution, training-state reconstruction. Proposed roadmap tracked in
`current.toml`; none of it exists here yet.

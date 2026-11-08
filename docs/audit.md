# Audit repairs and acceptance limits

Objective: prevent invalid checkpoint writes and unsupported runtime/training claims.
Publication remains on hold. No source checkpoint, deployed runtime or trained adapter
was changed by these repairs.

## Repaired

- MLX affine edits resolve per-module bits/group size, dequantize before editing,
  and regenerate packed weights, scales and biases together, including cross-shard triples.
- Output-space projection uses matrix rows and validates finite, nonzero directions.
- Block namespaces are discovered; ambiguous indexed stacks fail explicitly.
- Expert pruning validates fractions, permutations, bounds and complete MoE-layer coverage.
- Storage preserves every regular checkpoint asset recursively, verifies completion
  by default, detects inventory drift, and fails recovery for incomplete stores.
- Neuron edits, projection, expert pruning and DARE stream output groups. Input caches
  retain one shard per source; working memory still depends on shard/matrix size.
- DARE uses inverse-probability rescaling; TIES follows paper/mergekit (magnitude trim, summed-sign election, weighted disjoint mean). Hand-computed tests cover both.
- Corpus rows use structured tool calls and response IDs; training rejects legacy textual
  calls, unfilled rows and unsupported structured-training capability declarations.
- Runtime commands/endpoints/private aliases use hot-loaded TOML. Public code does not
  depend on a private backend name. Publication scans configured private literals/digests.
- Doctor counts nested methods/functions and checks tests as well as pipeline code.
- 2026-11-07 review: block discovery breaks text/vision/mtp ties by config `num_hidden_layers`
  (exact 0..N-1 coverage; equal depths still refuse), so VLM checkpoints resolve their text stack.
- 2026-11-07 review: training validation refuses `<tool_call>` chat-template text in assistant content.
- 2026-11-07 review: streamed writes pack groups into 2 GiB shards (`checkpoint/flush_shard.py`);
  previously one file per tensor group (1066 files for a 6-shard 35B-A3B source).

## Evidence scope

Tests exercise real CLI checkpoint writes with small synthetic MLX fixtures: 2/4/8-bit
neuron edits, mixed-bit DARE, expert pruning and projection, plus byte-identical storage
round trips, interruption/corruption refusal, private-name scanning and renamed executables.
These are structural/functional fixtures, not real-model quality or peak-memory proof.
The current publisher file selection and all six included images were reviewed for privacy.
Raw test/doctor/privacy reports remain in the private audit evidence directory.

## Explicit blocks

Dense-to-MoE conversion is limited to Qwen3 dense sources; routers are data-fitted, and
sparse top-K quality is measured, not assumed (docs/moefy.md). Profiling runs through runtime probe routes (docs/profile.md); in-process stubs stay fail-closed. Structured training is disabled
by default: the native dataset loader omits top-level tool schemas; no compatible trainer
was installed or run. Existing generated corpora/adapters remain unvalidated evidence.
Code/rev token grading and top-k KL remain diagnostic-only and cannot authorize promotion.
GGUF layer/expert pruning and GGUF ↔ safetensors conversion exist (docs/gguf.md); PyTorch editing
and dynamic runtime expert loading remain absent.

[MLX dequantization contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
and [TIES definition](https://arxiv.org/abs/2306.01708) informed the math review.

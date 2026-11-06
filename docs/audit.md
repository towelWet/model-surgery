# Code audit: verified changes and limits

Objective: improve the model-surgery CLI and its evidence checks before preparing
the local X preview. These changes do not establish a working full surgery campaign.

Fixed:

- Empty/report-only gate rules could authorize promotion without blocking evidence.
- Reports lacked a source fingerprint and comparable case-count checks.
- TOML strings were unescaped and float metrics were rounded to four decimals.
- One tool case was duplicated; another expected a tool absent from its schema.
- Origin sampling named fixed Qwen tensors/layers and ignored per-prefix quantization.
- BF16 row decoding discarded subnormals and mishandled special values.
- Depth pruning used an optimizable destination assertion, accepted invalid removal
  sets, and required optional `layer_types` configuration.

Extensions:

- `ms inspect MODEL [--out summary.toml]`: header-only checkpoint inspection;
  supports single-file/indexed layouts, checks byte bounds/index agreement.
- `ms cases [suite ...]`: offline case validation, counts and fingerprint.
- `ms list [stage ...] --search TEXT`: searchable, filesystem-discovered inventory.
- Bounded, discovered tensor sampling in the origin similarity tool.

Validation includes real depth-pruning CLI execution under `python -O` on a tiny
synthetic checkpoint, checking surviving tensor bytes. Fixtures are tests, not model
capability or profitability evidence. Full inference and healing were not run.

Remaining limitations:

- Per-layer residual/MLP taps and teacher-forced layer-skip profiling remain disabled;
  tensor-array instrumentation is still required. Doctor green does not prove profiling.
- Code/rev use token matching; KL uses renormalized top-k heads. Their reports are now
  explicitly diagnostic-only and cannot authorize promotion.
- Model labels in eval reports are caller supplied, not verified server identities.
- The archived SLAB baseline index references missing `optiq/optiq_vision.safetensors`;
  inspection correctly refuses it. Existing weights and indexes were not modified.
- Neuron editing, healing and other surgery paths were not end-to-end validated here.
- A source fingerprint establishes comparability, not authenticity of supplied reports.

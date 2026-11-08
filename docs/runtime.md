# Local runtime integration

Runtime selection uses protocol capabilities and private configuration, not product
name checks. Public defaults support tensor-array and Harn. A renamed compatible
executable needs only a configuration change; public code needs no private aliases.

Defaults live in `pipeline/runtime/defaults.toml`. Override them in the skill's
private `$CLAUDE_DIR/private/model-surgery.toml` (shape: `local.example.toml`) or a file selected by SURGERY_RUNTIME_CONFIG. Both use
the same runtime table, for example:

```toml
[runtime.http]
base_url = "http://localhost:8765/v1"

[runtime.commands.train]
candidates = [["custom-trainer"], ["another-compatible-trainer"]]

[runtime.commands.reload]
candidates = [["custom-runtime", "reload", "--model", "{model}"]]

[runtime.aliases]
model_path = ["CUSTOM_MODEL_VARIABLE", "SURGERY_MODEL"]

[runtime.environment]
CUSTOM_KERNEL_SETTING = "value"
```

Candidates are tried in order. Commands are argument arrays, never shell snippets.
Settings are read per request/resolution. SURGERY_TRAIN_COMMAND and SURGERY_API_URL
override selection; existing public environment variables remain compatible.
The reload wrapper delegates to the configured command; it does not kill processes,
force resident models, install services or change login settings.

`ms runtime` reports availability without printing private command names/paths.
`ms privacy` scans the publisher's exact file selection and the private forbidden
literal/digest policy. Private toml (outside the tree), evidence, generated corpora and research
are excluded from publication; Reference checkouts remain intact locally.

Structured training defaults to disabled. The current native dataset loader omits
top-level tool schemas. Set runtime.capabilities.structured_training only after the
adapter's schema/template/assistant-loss path is validated. A declared flag is not
inference or model-quality proof. Missing executables/capabilities fail explicitly.

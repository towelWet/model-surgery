# Stage examples

Set MODEL_SURGERY_ROOT and model locations through your own shell or private local.toml.
No workstation paths ship in this skill.

```sh
cd "$MODEL_SURGERY_ROOT"
python3 pipeline/cli/ms.py list
python3 pipeline/cli/ms.py inspect "$MODEL"
python3 pipeline/cli/ms.py cases
python3 pipeline/cli/ms.py doctor
python3 pipeline/surgery/depth/prune.py "$PARENT" "$CHILD" "$REMOVED_LAYERS"
python3 pipeline/cli/ms.py eval --out-dir "$REPORTS" --model "$MODEL_LABEL"
python3 pipeline/cli/ms.py gate --base "$BASE_REPORTS" --cand "$REPORTS"
```

Surgery writes new checkpoints. Inspect the source, choose a new destination and validate
profile evidence first. The current eval suites are diagnostic-only: the gate blocks them.
No example here proves model capability retention, inference speed, or a successful heal.

For a tensor-array server, configure the local endpoint with TARR_EVAL_PORT.
HOPE statistics require TARR_EMBEDDINGS and TARR_HOPE_STATS in that server's environment.
These examples do not install services or start a model.

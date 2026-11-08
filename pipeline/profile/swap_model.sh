#!/usr/bin/env bash
# Delegate explicit reload to the configured backend; never kill processes or manage login services.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/../runtime/reload.py" "$@"

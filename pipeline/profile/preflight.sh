#!/usr/bin/env bash
# Backend-independent checks only; starts nothing.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/../runtime/preflight.py" "$@"

#!/usr/bin/env bash
# run.sh — generic benchmark entry: exec check.py against exactly one config file.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "usage: run.sh <config.toml>" >&2                     # exactly one config filepath
    exit 2
fi
DIR="$(cd "$(dirname "$0")" && pwd)"                          # this script's folder
exec python3 "$DIR/check.py" "$1"

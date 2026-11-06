#!/usr/bin/env bash
# report.sh — run current profile preflight checks and propagate its exit status.
set -u
PF="$(cd "$(dirname "$0")/.." && pwd)/preflight.sh"           # preflight lives one level up
exec bash "$PF"

#!/usr/bin/env bash
# gpu-around-hotenv.sh — transition-only setenv on live :8766 (no model reload).
# $1 = metal (share, TARR_USE_METAL=1) | cpu (proven contention only). No sudo.
set -euo pipefail
want=${1:-metal}
val=1
[ "$want" = cpu ] && val=0
pid=$(lsof -nP -iTCP:8766 -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
[ -n "${pid:-}" ] || exit 0
log="${TMPDIR:-/tmp}/gpu-around-lldb.log"
lldb -p "$pid" --batch -o "expr -- (int)setenv(\"TARR_USE_METAL\", \"$val\", 1)" \
  -o detach -o quit >"$log" 2>&1 &
lp=$!
for _ in 1 2 3 4 5 6 7 8; do
  if ! kill -0 "$lp" 2>/dev/null; then wait "$lp" || true; exit 0; fi
  sleep 1
done
kill "$lp" 2>/dev/null || true
wait "$lp" 2>/dev/null || true
exit 0

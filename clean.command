#!/bin/zsh
# Remove generated Python caches throughout this skill, without following symlinks.
set -euo pipefail

skill_root="${0:A:h}"
cd -- "$skill_root"

/usr/bin/find -P "$skill_root" -type d -name '__pycache__' -prune \
    -exec /bin/rm -rf -- {} +
/usr/bin/find -P "$skill_root" -type f \( -name '*.pyc' -o -name '*.pyo' \) \
    -exec /bin/rm -f -- {} +

print -r -- "Python caches cleaned."

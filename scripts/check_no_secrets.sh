#!/usr/bin/env bash
# Refuses host paths, usernames and keys. This repo is public and a colleague's
# home directory has reached commit-ready state four times (build_asr_fragment_root.py,
# build_qazaqstantv_recollect_root.py, audit_report.py, build_qazaq_viewer.py),
# each time as a default or a docstring. Use $QAZ_RECOLLECT / $ASAN_ROOT instead.
#
#   bash scripts/check_no_secrets.sh              # staged files (use as a pre-commit hook)
#   bash scripts/check_no_secrets.sh --all        # every tracked file
#
# Install as a hook:
#   printf '#!/bin/sh\nexec bash scripts/check_no_secrets.sh\n' > .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

PATTERN='zhoshy|adilet_tasbolat|srp_2026|100\.85\.[0-9]+\.[0-9]+|10\.201\.[0-9]+\.[0-9]+|TRX50|/data/home/[a-z]|ghp_[A-Za-z0-9]{20}|sk-[A-Za-z0-9]{20}|BEGIN [A-Z ]*PRIVATE KEY'

if [ "${1:-}" = "--all" ]; then
  files=$(git ls-files)
else
  files=$(git diff --cached --name-only --diff-filter=ACM)
fi
[ -z "$files" ] && { echo "nothing to check"; exit 0; }

# Exclusions: the intentional placeholder in configs/config.yaml, and this
# script itself, whose own pattern string necessarily contains the strings.
files=$(printf '%s\n' $files | grep -v '^scripts/check_no_secrets.sh$' || true)
[ -z "$files" ] && { echo "ok: nothing to check"; exit 0; }
hits=$(grep -InE "$PATTERN" $files 2>/dev/null | grep -v '/data/home/<user>' || true)
if [ -n "$hits" ]; then
  echo "BLOCKED: host paths or identifiers found in files about to be committed:"
  echo "$hits"
  echo
  echo "Replace with an env var (QAZ_RECOLLECT, ASAN_ROOT) or a placeholder."
  exit 1
fi
echo "ok: no host paths or identifiers in $(echo "$files" | wc -l | xargs) file(s)"

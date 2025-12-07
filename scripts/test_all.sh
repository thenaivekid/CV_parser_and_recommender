#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate

export PYTHONPATH=.


echo "Running pytest on the tests/ directory..."

# Run pytest once for the entire tests/ directory. This is concise and professional.
if pytest -q tests; then
  echo "\nAll tests passed ✅"
  exit 0
else
  rc=$?
  echo "\nSome tests failed. See pytest output above. Exit code: ${rc}"
  exit ${rc}
fi

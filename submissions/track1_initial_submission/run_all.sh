#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for task_dir in "$ROOT_DIR"/task*; do
  echo "== Running ${task_dir##*/} =="
  bash "$task_dir/run.sh" "$@"
  echo
done

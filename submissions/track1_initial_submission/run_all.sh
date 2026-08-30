#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for task_dir in "$ROOT_DIR"/task*; do
  [[ -d "$task_dir" ]] || continue
  echo "== Running ${task_dir##*/} =="
  bash "$ROOT_DIR/run_task.sh" "${task_dir##*/}" "$@"
  echo
done

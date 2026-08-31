#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="${1:-}"

if [[ -z "$TASK_NAME" ]]; then
  echo "usage: bash run_task.sh <task-directory> [auto_bench arguments]" >&2
  exit 2
fi
shift

case "$TASK_NAME" in
  task[0-9][0-9]_*) ;;
  *)
    echo "invalid task directory: $TASK_NAME" >&2
    exit 2
    ;;
esac

TASK_DIR="$ROOT_DIR/$TASK_NAME"
if [[ ! -d "$TASK_DIR" ]]; then
  echo "task directory not found: $TASK_NAME" >&2
  exit 2
fi
if [[ ! -f "$TASK_DIR/base.py" ]]; then
  echo "missing reference implementation: $TASK_NAME/base.py" >&2
  exit 2
fi
if [[ ! -f "$TASK_DIR/submission.py" ]]; then
  echo "missing submission entry: $TASK_NAME/submission.py" >&2
  exit 2
fi

python "$ROOT_DIR/auto_bench.py" \
  --v0_file "$TASK_DIR/base.py" \
  --v1_file "$TASK_DIR/submission.py" \
  --warmup "${WARMUP:-50}" \
  --repeat "${REPEAT:-100}" \
  "$@"

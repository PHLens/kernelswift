#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
python "$ROOT_DIR/auto_bench.py"       --v0_file "$SCRIPT_DIR/base.py"       --v1_file "$SCRIPT_DIR/submission.py"       --warmup "${WARMUP:-50}"       --repeat "${REPEAT:-100}"       "$@"

#!/usr/bin/env python3
"""Build the disposable CoreX clock helper after source verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
LIB_ROOT = EXPERIMENT_ROOT / "lib"
sys.path.insert(0, str(LIB_ROOT))

from corex_clock import ClockHelperBuildError, build_corex_clock
from source_guard import SourceGuardError, verify_accepted_sources

DEFAULT_SOURCE = EXPERIMENT_ROOT / "device" / "corex_clock.cu"
DEFAULT_COREX_ROOT = Path(os.environ.get("COREX_ROOT", "/usr/local/corex-4.4.0"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corex-root", type=Path, default=DEFAULT_COREX_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    verify_accepted_sources(REPO_ROOT)
    return build_corex_clock(
        args.corex_root,
        args.source,
        args.output_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (SourceGuardError, ClockHelperBuildError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from catalog import (
    GENERATED_OUTPUT_PATHS,
    assert_generated_outputs_current,
    build_generated_outputs,
    write_generated_outputs,
)
from kernelwiki_common import KernelWikiError, run_cli
from validate import validate_skill_root


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root
    if root.is_symlink():
        raise KernelWikiError("generated-output-invalid", "generated root must not be a symlink", root)
    corpus = validate_skill_root(root, check_generated=False)
    outputs = build_generated_outputs(corpus)
    if args.check:
        assert_generated_outputs_current(root, outputs)
        payload = {"checked": len(GENERATED_OUTPUT_PATHS), "root": ".", "schema_version": 1}
    else:
        write_generated_outputs(root, outputs)
        payload = {"generated": len(GENERATED_OUTPUT_PATHS), "root": ".", "schema_version": 1}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from kernelwiki_common import require_within, run_cli
from provenance import (
    validate_provenance_skill_root,
    load_provenance,
    validate_provenance,
    validate_size_budget,
)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    root = validate_provenance_skill_root(args.root)
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = require_within(root, manifest)
    bundle = load_provenance(manifest)
    validate_provenance(bundle, root)
    validate_size_budget(root)
    print(
        json.dumps(
            {
                "asset_mode": bundle.asset_mode,
                "files": len(bundle.files),
                "schema_version": 1,
                "valid": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))

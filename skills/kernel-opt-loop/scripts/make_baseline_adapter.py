#!/usr/bin/env python3
"""Generate an immutable-base adapter by renaming Model to ModelNew."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
from typing import Sequence


class BaselineAdapterError(ValueError):
    """Raised when a baseline adapter cannot be generated safely."""


def find_model_class(tree: ast.Module) -> ast.ClassDef:
    """Return the sole top-level class named Model."""
    models = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Model"
    ]
    if len(models) != 1:
        raise BaselineAdapterError(
            f"expected exactly one top-level Model class, found {len(models)}"
        )
    return models[0]


def make_baseline_adapter(
    source: Path, destination: Path, force: bool = False
) -> None:
    """Write destination with the source's sole top-level Model renamed."""
    source = Path(source)
    destination = Path(destination)

    if source.resolve() == destination.resolve():
        raise BaselineAdapterError("source and destination must be different paths")
    if destination.exists() and not force:
        raise BaselineAdapterError(f"destination already exists: {destination}")

    try:
        source_text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BaselineAdapterError(f"cannot read source {source}: {error}") from error

    try:
        tree = ast.parse(source_text, filename=str(source))
    except SyntaxError as error:
        location = f"line {error.lineno}" if error.lineno is not None else "unknown line"
        raise BaselineAdapterError(
            f"cannot parse source {source} at {location}: {error.msg}"
        ) from error

    model = find_model_class(tree)
    model.name = "ModelNew"
    ast.fix_missing_locations(tree)

    try:
        destination.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BaselineAdapterError(
            f"cannot write destination {destination}: {error}"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rename one top-level Model class to ModelNew in a copied module."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing destination"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        make_baseline_adapter(args.source, args.destination, force=args.force)
    except BaselineAdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

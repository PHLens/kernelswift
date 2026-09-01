#!/usr/bin/env python3
"""Verify the curated BI150 Route C evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = EXPERIMENT_ROOT / "evidence-archive"
MANIFEST_NAME = "manifest.json"
SOURCE_COMMIT_SEMANTICS = (
    "Repository-derived files use the full Git SHA of their source tree; "
    "curated or installed-distribution files use a stable origin/version label."
)


class EvidenceArchiveError(ValueError):
    """Raised when the tracked evidence archive is incomplete or modified."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceArchiveError(f"{label} must be an object")
    return value


def _normalized_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceArchiveError(f"{label} must be a non-empty relative path")
    path = Path(value)
    drive_like = len(value) >= 3 and value[0].isalpha() and value[1:3] in (":/", ":\\")
    if (
        path.is_absolute()
        or drive_like
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
        or value == "."
    ):
        raise EvidenceArchiveError(f"{label} is not a normalized relative path: {value!r}")
    return value


def _safe_archive_path(value: object, label: str) -> str:
    relative = _normalized_relative_path(value, label)
    if relative == MANIFEST_NAME:
        raise EvidenceArchiveError("manifest.json must not recursively list itself")
    return relative


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceArchiveError(f"{label} must be a non-empty string")
    return value


def verify_evidence_archive(archive_root: Path = DEFAULT_ARCHIVE_ROOT) -> dict[str, Any]:
    root = Path(archive_root).resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise EvidenceArchiveError(f"manifest missing: {manifest_path}")

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    manifest = _mapping(manifest, "manifest")
    canonical_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_text != canonical_manifest:
        raise EvidenceArchiveError("manifest.json is not in deterministic canonical form")
    if manifest.get("document_type") != "bi150-route-c-evidence-archive-manifest":
        raise EvidenceArchiveError("unexpected manifest document_type")
    if manifest.get("schema_version") != 1:
        raise EvidenceArchiveError("unsupported manifest schema_version")
    if manifest.get("manifest_excludes") != [MANIFEST_NAME]:
        raise EvidenceArchiveError("manifest_excludes must contain only manifest.json")
    if manifest.get("source_commit_semantics") != SOURCE_COMMIT_SEMANTICS:
        raise EvidenceArchiveError(
            "source_commit_semantics must document Git SHA and stable origin/version labels"
        )

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise EvidenceArchiveError("manifest files must be an array")
    if manifest.get("file_count") != len(entries):
        raise EvidenceArchiveError("manifest file_count does not match files array")

    expected_paths: list[str] = []
    entry_by_path: dict[str, Mapping[str, Any]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"files[{index}]")
        relative = _safe_archive_path(entry.get("path"), f"files[{index}].path")
        if relative in entry_by_path:
            raise EvidenceArchiveError(f"duplicate manifest path: {relative}")
        _normalized_relative_path(entry.get("source_path"), f"{relative}.source_path")
        for field in ("source_commit", "source_root", "role", "status"):
            _require_nonempty_string(entry.get(field), f"{relative}.{field}")
        byte_count = entry.get("byte_count")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise EvidenceArchiveError(f"{relative}.byte_count must be a non-negative integer")
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise EvidenceArchiveError(f"{relative}.sha256 must be a lowercase SHA256")
        expected_paths.append(relative)
        entry_by_path[relative] = entry

    if expected_paths != sorted(expected_paths):
        raise EvidenceArchiveError("manifest files must be sorted by path")

    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )
    missing = sorted(set(expected_paths) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected_paths))
    if missing:
        raise EvidenceArchiveError("archive files missing: " + ", ".join(missing))
    if extra:
        raise EvidenceArchiveError("unexpected archive files: " + ", ".join(extra))

    forbidden = [
        path
        for path in actual_paths
        if path.endswith(".pyc") or "__pycache__" in Path(path).parts
    ]
    if forbidden:
        raise EvidenceArchiveError("forbidden Python bytecode: " + ", ".join(forbidden))

    total_bytes = 0
    for relative in expected_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise EvidenceArchiveError(f"archive entry is not a regular file: {relative}")
        entry = entry_by_path[relative]
        observed_size = path.stat().st_size
        if observed_size != entry["byte_count"]:
            raise EvidenceArchiveError(
                f"byte count mismatch for {relative}: "
                f"expected {entry['byte_count']}, observed {observed_size}"
            )
        observed_sha = sha256_file(path)
        if observed_sha != entry["sha256"]:
            raise EvidenceArchiveError(
                f"SHA256 mismatch for {relative}: "
                f"expected {entry['sha256']}, observed {observed_sha}"
            )
        total_bytes += observed_size

    return {
        "status": "valid",
        "archive_root": str(root),
        "file_count": len(expected_paths),
        "total_bytes": total_bytes,
        "manifest_sha256": sha256_file(manifest_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_evidence_archive(args.archive_root)
    except (EvidenceArchiveError, OSError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

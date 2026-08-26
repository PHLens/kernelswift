"""Shared vNext artifact loading, hashing, reference, and atomic-write helpers.

All vNext validators and the profile-probe runner import only these helpers so
no component duplicates path traversal, exclusive-create, atomic-write, or hash
logic. Public helpers never leak raw ``OSError``, ``UnicodeDecodeError``, or
``JSONDecodeError`` exceptions; malformed artifacts surface as stable
:class:`ContractValidationError` instances.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


class ContractValidationError(ValueError):
    """A stable, code-addressable vNext contract validation failure."""

    def __init__(self, code: str, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path

    def __str__(self) -> str:
        location = f" {self.path}" if self.path is not None else ""
        return f"{self.code}{location}: {self.message}"


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ContractValidationError("artifact-read", f"cannot read {path}", path) from error


def sha256_canonical_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_document(path: Path, *, artifact: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ContractValidationError("artifact-read", f"cannot read {artifact}", path) from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ContractValidationError("artifact-json-invalid", f"{artifact} is not valid JSON", path) from error
    if not isinstance(value, dict):
        raise ContractValidationError("artifact-object-required", f"{artifact} must be a JSON object", path)
    return value


def load_json_yaml_document(path: Path, *, artifact: str) -> dict[str, Any]:
    try:
        return load_json_document(path, artifact=artifact)
    except ContractValidationError as error:
        raise ContractValidationError(error.code, f"{artifact} must use json-compatible YAML: {error.message}", path) from error


def require_relative_artifact(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ContractValidationError("artifact-reference-invalid", "relative artifact reference is required")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or ".." in pure.parts:
        raise ContractValidationError("artifact-reference-invalid", "relative artifact reference must remain under project root")
    root = root.resolve()
    candidate = (root / Path(*pure.parts)).resolve()
    if root not in (candidate, *candidate.parents) or not candidate.is_file():
        raise ContractValidationError("artifact-reference-invalid", "relative artifact reference must name an existing file")
    return candidate


def validate_source_span(source: str, span: Mapping[str, Any]) -> None:
    start, end = span.get("start"), span.get("end")
    if not all(isinstance(point, list) and len(point) == 2 for point in (start, end)):
        raise ContractValidationError("source-span-invalid", "source span must contain start and end [line, column]")
    if any(isinstance(value, bool) or not isinstance(value, int) for point in (start, end) for value in point):
        raise ContractValidationError("source-span-invalid", "source span coordinates must be integers")
    lines = source.splitlines(keepends=True)
    if start[0] < 1 or end[0] < start[0] or end[0] > len(lines):
        raise ContractValidationError("source-span-invalid", "source span line range is invalid")
    if start[1] < 1 or end[1] < 1 or (end[0], end[1]) <= (start[0], start[1]):
        raise ContractValidationError("source-span-invalid", "source span must be a nonempty range")
    if start[1] > len(lines[start[0] - 1]) + 1 or end[1] > len(lines[end[0] - 1]) + 1:
        raise ContractValidationError("source-span-invalid", "source span column is outside source text")


def create_exclusive_directory(path: Path) -> Path:
    """Create *path* exclusively; a pre-existing directory is a stable error."""
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ContractValidationError("artifact-exclusive-create", f"refusing to reuse existing run directory {path}", path) from error
    except OSError as error:
        raise ContractValidationError("artifact-directory-error", f"cannot create run directory {path}", path) from error
    return path


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write *value* as sorted JSON plus a trailing newline.

    The temporary file lives in the same directory as *path* so ``Path.replace``
    is atomic on the same filesystem. Probe code must never partially overwrite
    a prior run's artifact.
    """
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as error:
        raise ContractValidationError("artifact-write-error", f"cannot write artifact {path}", path) from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any

import yaml


class KernelWikiError(Exception):
    def __init__(self, code: str, message: str, path: Path | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.path = path


def load_yaml_document(path: Path) -> Any:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise KernelWikiError("yaml-invalid", str(error), Path(path)) from error


def parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise KernelWikiError("frontmatter-missing", "Markdown must begin with ---", Path(path))
    marker = text.find("\n---\n", 4)
    if marker < 0:
        raise KernelWikiError("frontmatter-unclosed", "Markdown frontmatter is not closed", Path(path))
    metadata_text = text[4:marker]
    try:
        metadata = yaml.safe_load(metadata_text)
    except yaml.YAMLError as error:
        raise KernelWikiError("frontmatter-invalid", str(error), Path(path)) from error
    if not isinstance(metadata, dict):
        raise KernelWikiError("frontmatter-object-required", "frontmatter must be an object", Path(path))
    return metadata, text[marker + 5 :]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def require_within(root: Path, path: Path) -> Path:
    root = Path(root).resolve()
    candidate = Path(path).resolve()
    if candidate != root and root not in candidate.parents:
        raise KernelWikiError("path-escape", f"{candidate} escapes {root}", candidate)
    return candidate


def validate_root_relative_posix_path(value: Any) -> str:
    """Return a canonical root-relative POSIX path or raise ValueError."""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("path must be nonempty trimmed text")
    if "\\" in value or "\x00" in value:
        raise ValueError("path must use POSIX separators and contain no NUL")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise ValueError("path must be normalized and root-relative")
    return value


def parse_huawei_cann_document_revision(url: Any) -> str | None:
    """Return the immutable CANN revision encoded in an official document URL."""
    if not isinstance(url, str) or not url or url.strip() != url or "%" in url:
        return None
    match = re.fullmatch(
        r"https://www\.hiascend\.com/document/detail/[^/?#]+/"
        r"CANNCommunityEdition/(?P<revision>[A-Za-z0-9][A-Za-z0-9._-]*)/"
        r"[^?#]*[^/?#](?:#1)?",
        url,
    )
    return None if match is None else match.group("revision")


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_cli(main: Callable[[Sequence[str]], int], argv: Sequence[str] | None = None) -> int:
    try:
        return main(list(sys.argv[1:] if argv is None else argv))
    except KernelWikiError as error:
        location = f" ({error.path})" if error.path else ""
        print(f"error[{error.code}]: {error.message}{location}", file=sys.stderr)
        return 2

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import os
import re
import stat
from typing import Any

from kernelwiki_common import (
    KernelWikiError,
    load_yaml_document,
    parse_huawei_cann_document_revision,
    require_within,
    sha256_file,
)


PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "origin_url",
        "upstream_repo",
        "upstream_sha",
        "license_state",
        "retrieved_at",
        "asset_mode",
        "allowed_audiences",
        "coder_access",
        "source_ids",
        "files",
    }
)
PROVENANCE_FILE_FIELDS = frozenset(
    {"local_path", "upstream_path", "heading_path", "role", "mode", "sha256"}
)
ASSET_MODES = frozenset({"verbatim", "extracted", "derived"})
FILE_MODES = frozenset({"verbatim", "extracted", "derived", "upstream-patch"})
LICENSE_STATES = frozenset({"approved", "incompatible", "metadata-only", "unknown"})
AUDIENCES = frozenset({"coder", "designer"})
CODER_ACCESS_VALUES = frozenset({"denied", "snippet-only", "exact-profile"})
FILE_ROLES = frozenset(
    {"pr-diff", "upstream-file", "snippet", "historical-candidate", "bench-record"}
)
CODE_ROLES = frozenset({"pr-diff", "upstream-file", "snippet", "historical-candidate"})
ASSET_MODE_FILE_MODES = {
    "verbatim": frozenset({"verbatim", "upstream-patch"}),
    "extracted": frozenset({"extracted"}),
    "derived": frozenset({"derived"}),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SIZE_BUDGET_FIELDS = frozenset(
    {"schema_version", "repository_max_bytes", "bundle_max_bytes", "file_max_bytes"}
)


@dataclass(frozen=True)
class ProvenanceFile:
    local_path: str
    upstream_path: str | None
    heading_path: str | None
    role: str
    mode: str
    sha256: str


@dataclass(frozen=True)
class ProvenanceBundle:
    schema_version: int
    path: Path
    origin_url: str
    upstream_repo: str | None
    upstream_sha: str | None
    license_state: str
    retrieved_at: str
    asset_mode: str
    allowed_audiences: tuple[str, ...]
    coder_access: str
    source_ids: tuple[str, ...]
    files: tuple[ProvenanceFile, ...]


def _fail(code: str, message: str, path: Path) -> None:
    raise KernelWikiError(code, message, path)


def validate_provenance_skill_root(value: Path) -> Path:
    if not isinstance(value, Path):
        raise KernelWikiError(
            "provenance-path-invalid", "skill_root must be a pathlib.Path"
        )
    requested = value
    try:
        entry = os.lstat(requested)
    except (OSError, ValueError) as error:
        raise KernelWikiError(
            "provenance-path-invalid", "skill_root must be an existing real directory", requested
        ) from error
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise KernelWikiError(
            "provenance-path-invalid", "skill_root must be a real non-symlink directory", requested
        )
    try:
        return requested.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError(
            "provenance-path-invalid", "skill_root could not be resolved safely", requested
        ) from error


def _mapping(value: Any, code: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code, "expected a mapping", path)
    return value


def _closed_fields(value: Mapping[str, Any], expected: frozenset[str], prefix: str, path: Path) -> None:
    missing = sorted(expected - value.keys())
    if missing:
        _fail(f"{prefix}-field-required", f"missing fields: {', '.join(missing)}", path)
    unknown = sorted(value.keys() - expected)
    if unknown:
        _fail(f"{prefix}-field-unknown", f"unknown fields: {', '.join(unknown)}", path)


def _string(value: Any, code: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code, "expected a nonempty string", path)
    return value


def _nullable_string(value: Any, code: str, path: Path) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        _fail(code, "expected a nonempty string or null", path)
    return value


def _sorted_unique_strings(
    value: Any, code: str, path: Path, *, nonempty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(code, "expected a list", path)
    if nonempty and not value:
        _fail(code, "list must not be empty", path)
    if any(not isinstance(item, str) or not item.strip() for item in value):
        _fail(code, "list items must be nonempty strings", path)
    if value != sorted(set(value)):
        _fail(code, "list must be sorted and unique", path)
    return tuple(value)


def _enum(value: Any, allowed: frozenset[str], code: str, path: Path) -> str:
    value = _string(value, code, path)
    if value not in allowed:
        _fail(code, f"unsupported value {value!r}", path)
    return value


def load_provenance(path: Path) -> ProvenanceBundle:
    path = Path(path)
    document = _mapping(load_yaml_document(path), "provenance-invalid", path)
    _closed_fields(document, PROVENANCE_FIELDS, "provenance", path)
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        _fail("provenance-schema-invalid", "schema_version must be integer 1", path)

    origin_url = _string(document["origin_url"], "provenance-origin-invalid", path)
    upstream_repo = _nullable_string(document["upstream_repo"], "provenance-upstream-invalid", path)
    upstream_sha = _nullable_string(document["upstream_sha"], "provenance-upstream-invalid", path)
    official_revision = parse_huawei_cann_document_revision(origin_url)
    if official_revision is not None or upstream_repo == "hiascend.com/CANNCommunityEdition":
        if (
            official_revision is None
            or upstream_repo != "hiascend.com/CANNCommunityEdition"
            or upstream_sha != official_revision
        ):
            _fail(
                "provenance-upstream-invalid",
                "official Huawei provenance must bind upstream_sha to the immutable URL revision",
                path,
            )
    elif upstream_sha is not None and not GIT_SHA_RE.fullmatch(upstream_sha):
        _fail(
            "provenance-upstream-invalid",
            "upstream_sha must be a full lowercase Git SHA",
            path,
        )
    license_state = _enum(document["license_state"], LICENSE_STATES, "provenance-license-invalid", path)
    retrieved_at = _string(document["retrieved_at"], "provenance-retrieved-at-invalid", path)
    if not TIMESTAMP_RE.fullmatch(retrieved_at):
        _fail("provenance-retrieved-at-invalid", "retrieved_at must be UTC YYYY-MM-DDTHH:MM:SSZ", path)
    asset_mode = _enum(document["asset_mode"], ASSET_MODES, "provenance-mode-invalid", path)
    allowed_audiences = _sorted_unique_strings(
        document["allowed_audiences"], "provenance-audiences-invalid", path, nonempty=True
    )
    if not set(allowed_audiences) <= AUDIENCES:
        _fail("provenance-audiences-invalid", "allowed_audiences contains an unknown role", path)
    coder_access = _enum(document["coder_access"], CODER_ACCESS_VALUES, "provenance-coder-access-invalid", path)
    source_ids = _sorted_unique_strings(document["source_ids"], "provenance-source-ids-invalid", path)
    if any(not SOURCE_ID_RE.fullmatch(source_id) for source_id in source_ids):
        _fail("provenance-source-ids-invalid", "source_ids contains an invalid ID", path)

    raw_files = document["files"]
    if not isinstance(raw_files, list) or not raw_files:
        _fail("provenance-files-invalid", "files must be a nonempty list", path)
    files: list[ProvenanceFile] = []
    for raw_file in raw_files:
        item = _mapping(raw_file, "provenance-file-invalid", path)
        _closed_fields(item, PROVENANCE_FILE_FIELDS, "provenance-file", path)
        local_path = _string(item["local_path"], "provenance-local-path-invalid", path)
        upstream_path = _nullable_string(item["upstream_path"], "provenance-upstream-path-invalid", path)
        heading_path = _nullable_string(item["heading_path"], "provenance-heading-path-invalid", path)
        role = _enum(item["role"], FILE_ROLES, "provenance-role-invalid", path)
        mode = _enum(item["mode"], FILE_MODES, "provenance-file-mode-invalid", path)
        digest = _string(item["sha256"], "provenance-sha-invalid", path)
        if not SHA256_RE.fullmatch(digest):
            _fail("provenance-sha-invalid", "sha256 must be 64 lowercase hexadecimal characters", path)
        files.append(ProvenanceFile(local_path, upstream_path, heading_path, role, mode, digest))

    local_paths = [item.local_path for item in files]
    if local_paths != sorted(set(local_paths)):
        _fail("provenance-files-order", "files must be sorted and unique by local_path", path)

    return ProvenanceBundle(
        schema_version=1,
        path=path,
        origin_url=origin_url,
        upstream_repo=upstream_repo,
        upstream_sha=upstream_sha,
        license_state=license_state,
        retrieved_at=retrieved_at,
        asset_mode=asset_mode,
        allowed_audiences=allowed_audiences,
        coder_access=coder_access,
        source_ids=source_ids,
        files=tuple(files),
    )


def _validate_local_path(bundle_root: Path, local_path: str, manifest_path: Path) -> Path:
    pure = PurePosixPath(local_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("provenance-path-escape", f"invalid local_path {local_path!r}", manifest_path)
    if "\\" in local_path or pure.as_posix() == "PROVENANCE.yaml":
        _fail("provenance-path-escape", f"invalid local_path {local_path!r}", manifest_path)
    try:
        return require_within(bundle_root, bundle_root / Path(*pure.parts))
    except KernelWikiError as error:
        raise KernelWikiError("provenance-path-escape", error.message, manifest_path) from error


def _require_upstream(bundle: ProvenanceBundle) -> None:
    if bundle.upstream_repo is None or bundle.upstream_sha is None:
        _fail(
            "provenance-upstream-required",
            f"{bundle.asset_mode} evidence requires upstream_repo and upstream_sha",
            bundle.path,
        )


def _validate_mode_rules(bundle: ProvenanceBundle) -> None:
    if bundle.asset_mode in {"verbatim", "extracted"}:
        _require_upstream(bundle)
    if bundle.asset_mode == "derived" and not bundle.source_ids:
        _fail("provenance-source-required", "derived evidence requires source_ids", bundle.path)

    allowed_file_modes = ASSET_MODE_FILE_MODES[bundle.asset_mode]
    for item in bundle.files:
        if item.mode not in allowed_file_modes:
            _fail(
                "provenance-mode-mismatch",
                f"asset_mode {bundle.asset_mode!r} does not allow file mode {item.mode!r}",
                bundle.path,
            )
        if item.mode in {"verbatim", "upstream-patch", "extracted"}:
            _require_upstream(bundle)
            if item.upstream_path is None:
                _fail("provenance-upstream-path-required", f"{item.mode} file requires upstream_path", bundle.path)
        if item.mode == "extracted" and item.heading_path is None:
            _fail("provenance-locator-required", "extracted file requires heading_path", bundle.path)
        if item.mode == "derived":
            if not bundle.source_ids:
                _fail("provenance-source-required", "derived file requires source_ids", bundle.path)
            if item.upstream_path is not None:
                _fail("provenance-derived-upstream", "derived file must not claim an upstream_path", bundle.path)


def _validate_license_rules(bundle: ProvenanceBundle) -> None:
    if bundle.coder_access == "denied":
        if "coder" in bundle.allowed_audiences:
            _fail("provenance-coder-audience", "denied coder_access cannot allow the coder audience", bundle.path)
    elif "coder" not in bundle.allowed_audiences:
        _fail("provenance-coder-audience", "coder access requires the coder audience", bundle.path)

    code_files = tuple(item for item in bundle.files if item.role in CODE_ROLES)
    if bundle.license_state != "approved" and (bundle.coder_access != "denied" or code_files):
        _fail("license-code-exposure", "unapproved licenses may not expose code assets", bundle.path)
    if bundle.coder_access == "snippet-only" and any(item.role != "snippet" for item in code_files):
        _fail("provenance-coder-access", "snippet-only access may expose only snippet code assets", bundle.path)


def validate_provenance(bundle: ProvenanceBundle, skill_root: Path) -> None:
    skill_root = validate_provenance_skill_root(skill_root)
    manifest_path = require_within(skill_root, bundle.path)
    if manifest_path.name != "PROVENANCE.yaml" or manifest_path.is_symlink() or not manifest_path.is_file():
        _fail("provenance-path-invalid", "bundle path must be a regular PROVENANCE.yaml", bundle.path)
    artifacts_root = (skill_root / "artifacts").resolve()
    try:
        require_within(artifacts_root, manifest_path)
    except KernelWikiError as error:
        raise KernelWikiError("provenance-path-invalid", error.message, bundle.path) from error
    bundle_root = manifest_path.parent

    _validate_mode_rules(bundle)
    _validate_license_rules(bundle)

    declared: dict[str, ProvenanceFile] = {}
    for item in bundle.files:
        _validate_local_path(bundle_root, item.local_path, bundle.path)
        declared[item.local_path] = item

    actual: dict[str, Path] = {}
    for candidate in sorted(bundle_root.rglob("*")):
        if candidate == manifest_path:
            continue
        if candidate.is_symlink():
            _fail("provenance-path-escape", f"symlink is forbidden: {candidate}", bundle.path)
        if candidate.is_file():
            require_within(bundle_root, candidate)
            relative = candidate.relative_to(bundle_root).as_posix()
            actual[relative] = candidate

    missing = sorted(set(declared) - set(actual))
    if missing:
        _fail("asset-file-missing", f"declared files are missing: {', '.join(missing)}", bundle.path)
    undeclared = sorted(set(actual) - set(declared))
    if undeclared:
        _fail("asset-file-undeclared", f"undeclared files are present: {', '.join(undeclared)}", bundle.path)

    for local_path, item in declared.items():
        measured = sha256_file(actual[local_path])
        if measured != item.sha256:
            _fail(
                "asset-hash-mismatch",
                f"{local_path}: measured={measured} declared={item.sha256}",
                bundle.path,
            )


def _load_size_budget(skill_root: Path) -> tuple[int, int, int]:
    path = Path(skill_root) / "data" / "size-budget.yaml"
    document = _mapping(load_yaml_document(path), "size-budget-invalid", path)
    _closed_fields(document, SIZE_BUDGET_FIELDS, "size-budget", path)
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        _fail("size-budget-invalid", "schema_version must be integer 1", path)
    values: list[int] = []
    for field in ("repository_max_bytes", "bundle_max_bytes", "file_max_bytes"):
        value = document[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail("size-budget-invalid", f"{field} must be a positive integer", path)
        values.append(value)
    return values[0], values[1], values[2]


def _artifact_files(artifacts_root: Path) -> tuple[Path, ...]:
    if not artifacts_root.exists():
        return ()
    files: list[Path] = []
    for candidate in sorted(artifacts_root.rglob("*")):
        if candidate.is_symlink():
            _fail("provenance-path-escape", f"symlink is forbidden: {candidate}", candidate)
        if candidate.is_file() and candidate.name != ".gitkeep":
            require_within(artifacts_root, candidate)
            files.append(candidate)
    return tuple(files)


def validate_size_budget(skill_root: Path) -> None:
    skill_root = validate_provenance_skill_root(skill_root)
    repository_limit, bundle_limit, file_limit = _load_size_budget(skill_root)
    artifacts_path = skill_root / "artifacts"
    if artifacts_path.is_symlink():
        _fail("provenance-path-escape", "artifacts root must not be a symlink", artifacts_path)
    try:
        artifacts_root = require_within(skill_root, artifacts_path)
    except KernelWikiError as error:
        raise KernelWikiError("provenance-path-escape", error.message, artifacts_path) from error
    all_files = _artifact_files(artifacts_root)

    for path in all_files:
        measured = path.stat().st_size
        if measured > file_limit:
            _fail(
                "size-budget-file",
                f"{path.relative_to(skill_root).as_posix()}: measured={measured} allowed={file_limit}",
                path,
            )

    manifests = tuple(path for path in all_files if path.name == "PROVENANCE.yaml")
    for manifest in manifests:
        bundle_files = _artifact_files(manifest.parent)
        measured = sum(path.stat().st_size for path in bundle_files)
        if measured > bundle_limit:
            _fail(
                "size-budget-bundle",
                f"{manifest.parent.relative_to(skill_root).as_posix()}: measured={measured} allowed={bundle_limit}",
                manifest,
            )

    measured_repository = sum(path.stat().st_size for path in all_files)
    if measured_repository > repository_limit:
        _fail(
            "size-budget-repository",
            f"artifacts: measured={measured_repository} allowed={repository_limit}",
            artifacts_root,
        )

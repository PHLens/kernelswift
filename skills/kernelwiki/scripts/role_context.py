from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any

from kernel_opt_bridge import (
    LoopContractIdentity,
    compute_loop_contract_identity,
    load_loop_module,
    load_loop_modules,
    parse_loop_contract_identity,
)
from kernelwiki_common import KernelWikiError, canonical_json_bytes, sha256_bytes


CONTEXT_SCHEMA_VERSION = 1
SUPPORTED_CONTRACT_VERSION = 3
ROLES = frozenset({"designer", "coder"})
PROFILE_STATUSES = frozenset({"missing", "partial", "complete"})
REQUIRED_AUTHORITY_ARTIFACTS = frozenset(
    {"profile", "runtime_snapshot", "project_claim", "project_document", "sketch", "decision"}
)
_RUNTIME_FIELDS = frozenset({"target_id", "implementation_profile_id", "triton_version", "device_arch"})
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9._-]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRITON_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_JSON_FENCE_RE = re.compile(r"```json[ \t]*\r?\n(.*?)\r?\n```", flags=re.DOTALL)
_PROJECT_RUNTIME_RE = re.compile(
    r"^## runtime-fingerprint[ \t]*\r?\n(?:[ \t]*\r?\n)*(.*?)(?=\r?\n## |\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
_VALIDATION_TOKEN = object()
_FULL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "role",
        "target_id",
        "implementation_profile_id",
        "implementation_profile_status",
        "runtime_fingerprint",
        "languages",
        "dtypes",
        "operator_tags",
        "kernel_types",
        "semantic_features",
        "shape_signature",
        "current_bottlenecks",
        "project_root",
        "artifacts",
        "guidance_bindings",
        "loop_contract_identity",
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "target_id",
        "implementation_profile_id",
        "implementation_profile_status",
        "languages",
        "dtypes",
        "operator_tags",
        "kernel_types",
        "semantic_features",
        "shape_signature",
        "current_bottlenecks",
    }
)


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class RoleQueryContext:
    schema_version: int
    contract_version: int | None
    role: str
    target_id: str
    implementation_profile_id: str | None
    implementation_profile_status: str
    runtime_fingerprint: str | None
    languages: tuple[str, ...]
    dtypes: tuple[str, ...]
    operator_tags: tuple[str, ...]
    kernel_types: tuple[str, ...]
    semantic_features: tuple[str, ...]
    shape_signature: Mapping[str, Any]
    current_bottlenecks: tuple[str, ...]
    project_root: Path | None
    artifacts: Mapping[str, ArtifactRef]
    guidance_bindings: Mapping[str, tuple[str, ...]]
    loop_contract_identity: LoopContractIdentity | None
    _validation_token: object | None = field(default=None, init=False, repr=False, compare=False)
    _validation_fingerprint: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.loop_contract_identity is not None:
            if not isinstance(self.loop_contract_identity, LoopContractIdentity):
                raise KernelWikiError("contract-identity-invalid", "loop_contract_identity has an invalid type")
            object.__setattr__(self, "loop_contract_identity", _freeze_loop_identity(self.loop_contract_identity))


@dataclass(frozen=True)
class AuthoritySnapshot:
    contract_version: int
    loop_contract_identity: LoopContractIdentity
    profile: Mapping[str, Any]
    project_claim: Mapping[str, Any]
    sketch_result: Mapping[str, Any]
    decision_result: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]
    _validation_token: object | None = field(default=None, init=False, repr=False, compare=False)
    _validation_fingerprint: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.loop_contract_identity, LoopContractIdentity):
            raise KernelWikiError("contract-identity-invalid", "loop_contract_identity has an invalid type")
        object.__setattr__(self, "loop_contract_identity", _freeze_loop_identity(self.loop_contract_identity))


@dataclass(frozen=True)
class _FrozenAuthorityProject:
    root: Path
    artifact_paths: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _freeze_loop_identity(identity: LoopContractIdentity) -> LoopContractIdentity:
    return LoopContractIdentity(
        repository_commit=identity.repository_commit,
        skill_tree_sha=identity.skill_tree_sha,
        validator_sha256=MappingProxyType(dict(sorted(identity.validator_sha256.items()))),
        schema_sha256=MappingProxyType(dict(sorted(identity.schema_sha256.items()))),
    )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_plain_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ArtifactRef):
        return {"path": value.path.as_posix(), "sha256": value.sha256}
    if isinstance(value, LoopContractIdentity):
        return {
            "repository_commit": value.repository_commit,
            "skill_tree_sha": value.skill_tree_sha,
            "validator_sha256": _plain_json(value.validator_sha256),
            "schema_sha256": _plain_json(value.schema_sha256),
        }
    return value


def _context_fingerprint(context: RoleQueryContext) -> str:
    payload = {
        "schema_version": context.schema_version,
        "contract_version": context.contract_version,
        "role": context.role,
        "target_id": context.target_id,
        "implementation_profile_id": context.implementation_profile_id,
        "implementation_profile_status": context.implementation_profile_status,
        "runtime_fingerprint": context.runtime_fingerprint,
        "languages": context.languages,
        "dtypes": context.dtypes,
        "operator_tags": context.operator_tags,
        "kernel_types": context.kernel_types,
        "semantic_features": context.semantic_features,
        "shape_signature": context.shape_signature,
        "current_bottlenecks": context.current_bottlenecks,
        "project_root": context.project_root,
        "artifacts": context.artifacts,
        "guidance_bindings": context.guidance_bindings,
        "loop_contract_identity": context.loop_contract_identity,
    }
    return sha256_bytes(canonical_json_bytes(_plain_json(payload)))


def _authority_fingerprint(authority: AuthoritySnapshot) -> str:
    payload = {
        "contract_version": authority.contract_version,
        "loop_contract_identity": authority.loop_contract_identity,
        "profile": authority.profile,
        "project_claim": authority.project_claim,
        "sketch_result": authority.sketch_result,
        "decision_result": authority.decision_result,
        "artifact_hashes": authority.artifact_hashes,
    }
    return sha256_bytes(canonical_json_bytes(_plain_json(payload)))


def _seal_role_context(context: RoleQueryContext) -> RoleQueryContext:
    object.__setattr__(context, "_validation_token", _VALIDATION_TOKEN)
    object.__setattr__(context, "_validation_fingerprint", _context_fingerprint(context))
    return context


def _seal_authority_snapshot(authority: AuthoritySnapshot) -> AuthoritySnapshot:
    object.__setattr__(authority, "_validation_token", _VALIDATION_TOKEN)
    object.__setattr__(authority, "_validation_fingerprint", _authority_fingerprint(authority))
    return authority


def require_validated_role_context(value: Any) -> RoleQueryContext:
    if type(value) is not RoleQueryContext or value._validation_token is not _VALIDATION_TOKEN:
        raise KernelWikiError("contract-unsupported", "context was not issued by load_role_context")
    try:
        current = _context_fingerprint(value)
        _validate_role_conditionals(value)
    except Exception as error:
        raise KernelWikiError("contract-unsupported", "validated context invariants are invalid") from error
    if value._validation_fingerprint != current:
        raise KernelWikiError("contract-unsupported", "validated context invariants changed after loading")
    return value


def require_validated_authority_snapshot(value: Any) -> AuthoritySnapshot:
    if type(value) is not AuthoritySnapshot or value._validation_token is not _VALIDATION_TOKEN:
        raise KernelWikiError("contract-unsupported", "authority was not issued by load_authority_snapshot")
    try:
        current = _authority_fingerprint(value)
    except Exception as error:
        raise KernelWikiError("contract-unsupported", "validated authority invariants are invalid") from error
    if value._validation_fingerprint != current:
        raise KernelWikiError("contract-unsupported", "validated authority invariants changed after loading")
    if value.contract_version != SUPPORTED_CONTRACT_VERSION:
        raise KernelWikiError("contract-unsupported", "authority contract version is unsupported")
    return value


def _decode_json_object(data: bytes, *, path: Path, code: str, label: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_strict_object_pairs)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise KernelWikiError(code, f"cannot load {label}: {error}", path) from error
    if not isinstance(value, dict):
        raise KernelWikiError(code, f"{label} must contain a JSON object", path)
    return value


def _read_json_object(path: Path, *, code: str, label: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise KernelWikiError(code, f"{label} must be a regular file", path)
        data = path.read_bytes()
    except KernelWikiError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError(code, f"cannot load {label}: {error}", path) from error
    return _decode_json_object(data, path=path, code=code, label=label)


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise KernelWikiError("role-context-invalid", f"{field} must be null or a nonempty trimmed string")
    return value


def _identifier(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise KernelWikiError("role-context-invalid", f"{field} must be a safe identifier")
    return value


def _string_tuple(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise KernelWikiError("role-context-invalid", f"{field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item.strip() != item:
            raise KernelWikiError("role-context-invalid", f"{field} must contain nonempty trimmed strings")
        if item in result:
            raise KernelWikiError("role-context-invalid", f"{field} must not contain duplicates")
        result.append(item)
    if not allow_empty and not result:
        raise KernelWikiError("role-context-invalid", f"{field} must not be empty")
    return tuple(result)


def _shape_value(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and (not value.strip() or value.strip() != value):
            raise KernelWikiError("role-context-invalid", f"{field} strings must be nonempty and trimmed")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise KernelWikiError("role-context-invalid", f"{field} numbers must be finite")
        return value
    if isinstance(value, list):
        return tuple(_shape_value(item, field) for item in value)
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip() or key.strip() != key:
                raise KernelWikiError("role-context-invalid", f"{field} keys must be nonempty trimmed strings")
            output[key] = _shape_value(item, field)
        return MappingProxyType(output)
    raise KernelWikiError("role-context-invalid", f"{field} contains an unsupported value")


def _shape_signature(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise KernelWikiError("role-context-invalid", "shape_signature must be an object")
    result = _shape_value(value, "shape_signature")
    if not isinstance(result, Mapping):
        raise KernelWikiError("role-context-invalid", "shape_signature must be an object")
    return result


def _relative_artifact_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or value.strip() != value or "\\" in value or "\x00" in value:
        raise KernelWikiError("artifact-path-invalid", f"artifact {field!r} path must be normalized POSIX text")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts) or pure.as_posix() != value:
        raise KernelWikiError("artifact-path-escape", f"artifact {field!r} path escapes the project root")
    return Path(*pure.parts)


def _artifact_refs(value: Any) -> Mapping[str, ArtifactRef]:
    if not isinstance(value, dict):
        raise KernelWikiError("role-context-invalid", "artifacts must be an object")
    result: dict[str, ArtifactRef] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or _IDENTIFIER_RE.fullmatch(name) is None:
            raise KernelWikiError("role-context-invalid", "artifact names must be safe identifiers")
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise KernelWikiError("role-context-invalid", f"artifact {name!r} must contain path and sha256")
        digest = raw["sha256"]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise KernelWikiError("artifact-sha-invalid", f"artifact {name!r} sha256 is malformed")
        result[name] = ArtifactRef(path=_relative_artifact_path(raw["path"], name), sha256=digest)
    return MappingProxyType(result)


def _guidance_bindings(value: Any) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise KernelWikiError("role-context-invalid", "guidance_bindings must be an object")
    output: dict[str, tuple[str, ...]] = {}
    for guidance_id, statement_ids in value.items():
        if not isinstance(guidance_id, str) or not guidance_id.strip() or guidance_id.strip() != guidance_id:
            raise KernelWikiError("role-context-invalid", "guidance IDs must be nonempty trimmed strings")
        output[guidance_id] = _string_tuple(statement_ids, f"guidance_bindings.{guidance_id}", allow_empty=False)
    return MappingProxyType(output)


def _project_root(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise KernelWikiError("project-root-invalid", "project_root must be null or an absolute directory path")
    try:
        path = Path(value)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise KernelWikiError("project-root-invalid", "project_root must be a real absolute directory", path)
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise KernelWikiError("project-root-invalid", "project_root must not traverse symlinks or aliases", path)
        return resolved
    except KernelWikiError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError("project-root-invalid", "project_root cannot be resolved") from error


def load_role_context(path: Path) -> RoleQueryContext:
    document = _read_json_object(Path(path), code="role-context-invalid", label="role query context")
    fields = set(document)
    if not _REQUIRED_FIELDS.issubset(fields) or not fields.issubset(_FULL_FIELDS):
        missing = sorted(_REQUIRED_FIELDS - fields)
        unknown = sorted(fields - _FULL_FIELDS)
        raise KernelWikiError("role-context-invalid", f"context fields invalid; missing={missing}, unknown={unknown}", Path(path))
    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != CONTEXT_SCHEMA_VERSION:
        raise KernelWikiError("role-context-invalid", "schema_version must be integer 1", Path(path))
    role = document["role"]
    if not isinstance(role, str) or role not in ROLES:
        raise KernelWikiError("role-context-invalid", "role must be designer or coder", Path(path))
    target_id = _identifier(document["target_id"], "target_id")
    profile_id = _identifier(document["implementation_profile_id"], "implementation_profile_id", optional=True)
    profile_status = document["implementation_profile_status"]
    if not isinstance(profile_status, str) or profile_status not in PROFILE_STATUSES:
        raise KernelWikiError("role-context-invalid", "implementation_profile_status is invalid")
    contract_version = document.get("contract_version")
    if contract_version is not None and (type(contract_version) is not int or contract_version != SUPPORTED_CONTRACT_VERSION):
        raise KernelWikiError("contract-unsupported", "only role contract version 3 is supported")
    runtime_fingerprint = _optional_string(document.get("runtime_fingerprint"), "runtime_fingerprint")
    root = _project_root(document.get("project_root"))
    artifacts = _artifact_refs(document.get("artifacts", {}))
    bindings = _guidance_bindings(document.get("guidance_bindings", {}))
    identity_raw = document.get("loop_contract_identity")
    identity = None if identity_raw is None else parse_loop_contract_identity(identity_raw)

    context = RoleQueryContext(
        schema_version=schema_version,
        contract_version=contract_version,
        role=role,
        target_id=str(target_id),
        implementation_profile_id=profile_id,
        implementation_profile_status=profile_status,
        runtime_fingerprint=runtime_fingerprint,
        languages=_string_tuple(document["languages"], "languages"),
        dtypes=_string_tuple(document["dtypes"], "dtypes"),
        operator_tags=_string_tuple(document["operator_tags"], "operator_tags"),
        kernel_types=_string_tuple(document["kernel_types"], "kernel_types"),
        semantic_features=_string_tuple(document["semantic_features"], "semantic_features"),
        shape_signature=_shape_signature(document["shape_signature"]),
        current_bottlenecks=_string_tuple(document["current_bottlenecks"], "current_bottlenecks"),
        project_root=root,
        artifacts=artifacts,
        guidance_bindings=bindings,
        loop_contract_identity=identity,
    )
    _validate_role_conditionals(context)
    return _seal_role_context(context)


def _validate_role_conditionals(context: RoleQueryContext) -> None:
    if context.role == "designer":
        if context.artifacts or context.guidance_bindings or context.project_root is not None or context.loop_contract_identity is not None:
            raise KernelWikiError("role-context-invalid", "Designer context cannot carry loop authority artifacts")
        return
    if context.contract_version != SUPPORTED_CONTRACT_VERSION:
        raise KernelWikiError("contract-unsupported", "Coder context requires contract version 3")
    if context.implementation_profile_id is None:
        raise KernelWikiError("profile-missing", "Coder context requires an implementation profile ID")
    if context.implementation_profile_status == "missing":
        if (
            context.runtime_fingerprint is not None
            or context.project_root is not None
            or context.artifacts
            or context.guidance_bindings
            or context.loop_contract_identity is not None
        ):
            raise KernelWikiError("role-context-invalid", "missing-profile Coder context cannot carry fallback authority")
        return
    if not context.dtypes:
        raise KernelWikiError("role-context-invalid", "non-missing Coder dtypes must not be empty")
    if context.runtime_fingerprint is None:
        raise KernelWikiError("role-context-invalid", "Coder runtime_fingerprint is required")
    if context.project_root is None:
        raise KernelWikiError("project-root-invalid", "Coder project_root is required")
    if set(context.artifacts) != REQUIRED_AUTHORITY_ARTIFACTS:
        raise KernelWikiError(
            "artifact-required",
            "Coder context requires profile, runtime_snapshot, project_claim, project_document, sketch, and decision",
        )
    if context.artifacts["project_document"].path != Path("project.md"):
        raise KernelWikiError("artifact-path-invalid", "project_document must pin project.md")
    if context.loop_contract_identity is None:
        raise KernelWikiError("contract-identity-invalid", "Coder context requires loop_contract_identity")


def _read_regular_file_at(parent_fd: int, name: str, display_path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise KernelWikiError("artifact-missing", "cannot open frozen project file", display_path) from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise KernelWikiError("artifact-path-escape", "frozen project entries must be regular files", display_path)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise KernelWikiError("artifact-hash-mismatch", "project file changed while being frozen", display_path)
        return b"".join(chunks)
    except KernelWikiError:
        raise
    except OSError as error:
        raise KernelWikiError("artifact-missing", "cannot read frozen project file", display_path) from error
    finally:
        os.close(fd)


def _freeze_directory(source_fd: int, destination: Path, relative: PurePosixPath, captured: dict[str, bytes]) -> None:
    try:
        names = sorted(os.listdir(source_fd))
    except OSError as error:
        raise KernelWikiError("artifact-missing", "cannot enumerate project_root") from error
    for name in names:
        if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\\" in name or "\x00" in name:
            raise KernelWikiError("artifact-path-escape", "project contains an invalid path component")
        child_relative = relative / name
        display_path = Path(child_relative.as_posix())
        try:
            entry = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError as error:
            raise KernelWikiError("artifact-missing", "cannot inspect project entry", display_path) from error
        destination_path = destination / name
        if stat.S_ISDIR(entry.st_mode):
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                child_fd = os.open(name, flags, dir_fd=source_fd)
            except OSError as error:
                raise KernelWikiError("artifact-path-escape", "project directory changed while being frozen", display_path) from error
            destination_path.mkdir(mode=0o700)
            try:
                _freeze_directory(child_fd, destination_path, child_relative, captured)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(entry.st_mode):
            data = _read_regular_file_at(source_fd, name, display_path)
            destination_path.write_bytes(data)
            captured[child_relative.as_posix()] = data
        else:
            raise KernelWikiError("artifact-path-escape", "project cannot contain symlinks or special files", display_path)


def _make_snapshot_read_only(root: Path) -> None:
    files = sorted((path for path in root.rglob("*") if path.is_file()), reverse=True)
    directories = sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True)
    for path in files:
        path.chmod(0o400)
    for path in directories:
        path.chmod(0o500)
    root.chmod(0o500)


def _make_snapshot_writable(root: Path) -> None:
    if not root.exists():
        return
    try:
        root.chmod(0o700)
    except OSError:
        return
    for path in root.rglob("*"):
        try:
            if path.is_dir():
                path.chmod(0o700)
            else:
                path.chmod(0o600)
        except OSError:
            pass


@contextmanager
def _freeze_authority_project(context: RoleQueryContext) -> Iterator[_FrozenAuthorityProject]:
    if context.project_root is None:
        raise KernelWikiError("project-root-invalid", "Coder project_root is missing")
    root_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(str(context.project_root), root_flags)
    except OSError as error:
        raise KernelWikiError("project-root-invalid", "cannot open project_root without following links", context.project_root) from error
    try:
        before = os.fstat(root_fd)
        with tempfile.TemporaryDirectory(prefix="kernelwiki-authority-project-") as directory:
            frozen_root = Path(directory) / "project"
            frozen_root.mkdir(mode=0o700)
            captured: dict[str, bytes] = {}
            try:
                _freeze_directory(root_fd, frozen_root, PurePosixPath(), captured)
                after = os.fstat(root_fd)
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise KernelWikiError("project-root-invalid", "project_root changed while being frozen", context.project_root)

                artifact_paths: dict[str, Path] = {}
                artifact_hashes: dict[str, str] = {}
                for name in sorted(REQUIRED_AUTHORITY_ARTIFACTS):
                    reference = context.artifacts[name]
                    relative = reference.path.as_posix()
                    data = captured.get(relative)
                    if data is None:
                        raise KernelWikiError("artifact-missing", f"artifact {name!r} is missing", context.project_root / reference.path)
                    actual = sha256_bytes(data)
                    if actual != reference.sha256:
                        raise KernelWikiError("artifact-hash-mismatch", f"artifact {name!r} hash does not match", context.project_root / reference.path)
                    artifact_paths[name] = frozen_root / reference.path
                    artifact_hashes[name] = actual

                _make_snapshot_read_only(frozen_root)
                yield _FrozenAuthorityProject(
                    root=frozen_root,
                    artifact_paths=MappingProxyType(dict(artifact_paths)),
                    artifact_hashes=MappingProxyType(dict(sorted(artifact_hashes.items()))),
                )
            finally:
                _make_snapshot_writable(frozen_root)
    finally:
        os.close(root_fd)


def _runtime_snapshot(path: Path) -> dict[str, str]:
    document = _read_json_object(path, code="authority-invalid", label="runtime snapshot")
    if set(document) != _RUNTIME_FIELDS:
        raise KernelWikiError("authority-invalid", "runtime snapshot must contain exactly four authority fields", path)
    output: dict[str, str] = {}
    for field in sorted(_RUNTIME_FIELDS):
        value = document[field]
        if not isinstance(value, str) or not value.strip() or value.strip() != value:
            raise KernelWikiError("authority-invalid", f"runtime snapshot {field} must be a nonempty trimmed string", path)
        output[field] = value
    if _TRITON_VERSION_RE.fullmatch(output["triton_version"]) is None:
        raise KernelWikiError("authority-invalid", "runtime snapshot triton_version must be semantic x.y.z", path)
    return output


def _validate_json_compatible_profile(path: Path) -> None:
    try:
        data = path.read_bytes()
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError("authority-invalid", f"cannot read implementation profile: {error}", path) from error
    if data.lstrip().startswith(b"{"):
        _decode_json_object(data, path=path, code="authority-invalid", label="implementation profile")


def _validate_decision_json_blocks(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, RuntimeError, ValueError) as error:
        raise KernelWikiError("authority-invalid", f"cannot read Decision JSON blocks: {error}", path) from error
    matches = tuple(_JSON_FENCE_RE.finditer(text))
    if not matches:
        raise KernelWikiError("authority-invalid", "Decision must contain fenced JSON blocks", path)
    for index, match in enumerate(matches, start=1):
        try:
            value = json.loads(match.group(1), object_pairs_hook=_strict_object_pairs)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise KernelWikiError(
                "authority-invalid",
                f"Decision JSON block {index} is invalid: {error}",
                path,
            ) from error
        if not isinstance(value, dict):
            raise KernelWikiError("authority-invalid", f"Decision JSON block {index} must be an object", path)


def _project_runtime_fingerprint(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, RuntimeError, ValueError) as error:
        raise KernelWikiError("authority-invalid", f"cannot read project runtime fingerprint: {error}", path) from error
    match = _PROJECT_RUNTIME_RE.search(text)
    if match is None:
        raise KernelWikiError("runtime-mismatch", "project.md lacks ## runtime-fingerprint", path)
    value = match.group(1).strip()
    if not value or "\n" in value or "\r" in value:
        raise KernelWikiError("runtime-mismatch", "project.md runtime fingerprint must be one nonempty line", path)
    return value


def _normalized_target_family(target_id: str, device_arch: str, path: Path) -> str:
    families: list[str] = []
    for value in (target_id, device_arch):
        match = re.match(r"[a-z]+", value.casefold())
        if match is None:
            raise KernelWikiError("authority-invalid", "runtime target family cannot be normalized", path)
        families.append(match.group(0))
    if families[0] != families[1]:
        raise KernelWikiError("authority-invalid", "runtime target and device architecture families differ", path)
    return families[0]


def _mapping(value: Any, label: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KernelWikiError("authority-invalid", f"{label} must be an object", path)
    return value


def _bind_profile_and_runtime(
    context: RoleQueryContext,
    profile: Mapping[str, Any],
    project_claim: Mapping[str, Any],
    runtime_snapshot: Mapping[str, str],
    paths: Mapping[str, Path],
) -> None:
    profile_path = paths["profile"]
    runtime_path = paths["runtime_snapshot"]
    claim_path = paths["project_claim"]
    if profile.get("implementation_profile_id") != context.implementation_profile_id:
        raise KernelWikiError("profile-version-mismatch", "loaded profile ID does not match context", profile_path)
    if profile.get("profile_status") != context.implementation_profile_status:
        raise KernelWikiError("profile-version-mismatch", "loaded profile status does not match context", profile_path)
    if runtime_snapshot["target_id"] != context.target_id:
        raise KernelWikiError("authority-invalid", "runtime snapshot target_id does not match context", runtime_path)
    if runtime_snapshot["implementation_profile_id"] != context.implementation_profile_id:
        raise KernelWikiError("profile-version-mismatch", "runtime snapshot profile does not match context", runtime_path)

    identity_match = _mapping(profile.get("identity_match"), "profile identity_match", profile_path)
    targets = identity_match.get("permitted_target_ids")
    arches = identity_match.get("permitted_device_architectures")
    if not isinstance(targets, list) or context.target_id not in targets:
        raise KernelWikiError("profile-version-mismatch", "context target is outside the profile identity", profile_path)
    if not isinstance(arches, list) or runtime_snapshot["device_arch"] not in arches:
        raise KernelWikiError("profile-version-mismatch", "runtime device architecture is outside the profile identity", profile_path)

    implementation = _mapping(profile.get("implementation"), "profile implementation", profile_path)
    backend = implementation.get("backend")
    toolchain = implementation.get("toolchain")
    if not isinstance(backend, str) or not backend.strip() or backend.strip() != backend:
        raise KernelWikiError("profile-version-mismatch", "profile implementation backend is invalid", profile_path)
    if not isinstance(toolchain, str) or not toolchain.strip() or toolchain.strip() != toolchain:
        raise KernelWikiError("runtime-mismatch", "profile implementation toolchain is invalid", profile_path)
    if context.languages != (backend,):
        raise KernelWikiError("profile-version-mismatch", "Coder languages must exactly match the profile backend", profile_path)

    claim = _mapping(project_claim.get("claim"), "validated project claim", claim_path)
    claim_runtime = claim.get("runtime_fingerprint")
    if not isinstance(context.runtime_fingerprint, str) or claim_runtime != context.runtime_fingerprint:
        raise KernelWikiError("runtime-mismatch", "project claim runtime does not match context", claim_path)
    if toolchain != context.runtime_fingerprint:
        raise KernelWikiError("runtime-mismatch", "profile toolchain does not match context runtime", profile_path)
    project_runtime = _project_runtime_fingerprint(paths["project_document"])
    if project_runtime != context.runtime_fingerprint:
        raise KernelWikiError("runtime-mismatch", "project.md runtime fingerprint does not match context", paths["project_document"])
    runtime_components = tuple(part.strip().casefold() for part in context.runtime_fingerprint.split("/"))
    if not runtime_components or any(not part for part in runtime_components):
        raise KernelWikiError("runtime-mismatch", "runtime fingerprint components are malformed", runtime_path)
    expected_component = f"{backend} {runtime_snapshot['triton_version']}".casefold()
    if expected_component not in runtime_components:
        raise KernelWikiError(
            "runtime-mismatch",
            "runtime fingerprint lacks an exact backend/version component",
            runtime_path,
        )


_DECISION_ARTIFACT_FIELDS = MappingProxyType(
    {
        "sketch": ("sketch_ref", "sketch_sha256"),
        "profile": ("implementation_profile_snapshot_ref", "implementation_profile_snapshot_sha256"),
        "project_claim": ("project_capability_claim_ref", "project_capability_claim_sha256"),
    }
)


def _bind_decision_artifact_refs(
    context: RoleQueryContext,
    decision_result: Mapping[str, Any],
    decision_path: Path,
) -> None:
    metadata = _mapping(decision_result.get("metadata"), "validated Decision metadata", decision_path)
    for artifact_name, (ref_field, sha_field) in _DECISION_ARTIFACT_FIELDS.items():
        expected = context.artifacts[artifact_name]
        for container, label in ((metadata, "metadata"), (decision_result, "result")):
            raw_ref = container.get(ref_field)
            try:
                normalized_ref = _relative_artifact_path(raw_ref, f"Decision {label}.{ref_field}")
            except KernelWikiError as error:
                raise KernelWikiError(
                    "authority-invalid",
                    f"Decision {label} {ref_field} is not a normalized project-relative path",
                    decision_path,
                ) from error
            if normalized_ref != expected.path:
                raise KernelWikiError(
                    "authority-invalid",
                    f"Decision {label} {ref_field} does not match context artifact {artifact_name}",
                    decision_path,
                )
            raw_sha = container.get(sha_field)
            if not isinstance(raw_sha, str) or _SHA256_RE.fullmatch(raw_sha) is None:
                raise KernelWikiError(
                    "authority-invalid",
                    f"Decision {label} {sha_field} is not a SHA-256 digest",
                    decision_path,
                )
            if raw_sha != expected.sha256:
                raise KernelWikiError(
                    "authority-invalid",
                    f"Decision {label} {sha_field} does not match context artifact {artifact_name}",
                    decision_path,
                )


def _bind_decision_metadata(
    context: RoleQueryContext,
    profile: Mapping[str, Any],
    runtime_snapshot: Mapping[str, str],
    decision_result: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    decision_path = paths["decision"]
    profile_path = paths["profile"]
    runtime_path = paths["runtime_snapshot"]
    metadata = _mapping(decision_result.get("metadata"), "validated Decision metadata", decision_path)
    implementation = _mapping(profile.get("implementation"), "profile implementation", profile_path)
    language = implementation.get("backend")
    if not isinstance(language, str) or context.languages != (language,):
        raise KernelWikiError("profile-version-mismatch", "Decision language authority is not bound to the profile", profile_path)
    if metadata.get("language") != language:
        raise KernelWikiError("profile-version-mismatch", "Decision language does not match the Coder/profile backend", decision_path)
    target_family = _normalized_target_family(context.target_id, runtime_snapshot["device_arch"], runtime_path)
    if metadata.get("backend") != target_family:
        raise KernelWikiError("profile-version-mismatch", "Decision backend does not match the runtime target family", decision_path)
    if metadata.get("runtime_fingerprint_ref") != "project.md#runtime-fingerprint":
        raise KernelWikiError("runtime-mismatch", "Decision runtime_fingerprint_ref must equal project.md#runtime-fingerprint", decision_path)


def _raise_authority_error(error: Exception) -> None:
    code = getattr(error, "code", "authority-invalid")
    message = getattr(error, "message", str(error))
    path = getattr(error, "path", None)
    raise KernelWikiError("authority-invalid", f"{code}: {message}", path if isinstance(path, Path) else None) from error


def load_authority_snapshot(context: RoleQueryContext) -> AuthoritySnapshot:
    context = require_validated_role_context(context)
    if context.role != "coder":
        raise KernelWikiError("authority-context-invalid", "authority snapshot requires a Coder RoleQueryContext")
    if context.implementation_profile_status == "missing":
        raise KernelWikiError("profile-missing", f"implementation profile {context.implementation_profile_id!r} is missing")
    if context.contract_version != SUPPORTED_CONTRACT_VERSION or context.loop_contract_identity is None:
        raise KernelWikiError("contract-unsupported", "Coder authority contract is unsupported")

    with load_loop_modules(("validate_profile", "validate_sketch", "validate_decision")) as (current_identity, modules):
        if context.loop_contract_identity != current_identity:
            raise KernelWikiError("contract-unsupported", "pinned loop contract identity does not match committed authority")
        with _freeze_authority_project(context) as frozen:
            paths = frozen.artifact_paths
            runtime_snapshot = _runtime_snapshot(paths["runtime_snapshot"])
            _validate_json_compatible_profile(paths["profile"])
            _read_json_object(paths["project_claim"], code="authority-invalid", label="project capability claim")
            _read_json_object(paths["sketch"], code="authority-invalid", label="Sketch")
            _validate_decision_json_blocks(paths["decision"])

            profile_module = modules["validate_profile"]
            sketch_module = modules["validate_sketch"]
            decision_module = modules["validate_decision"]
            try:
                profile = profile_module.load_profile(paths["profile"])
                project_claim = profile_module.validate_project_claim(
                    paths["project_claim"],
                    profile=profile,
                    snapshot=runtime_snapshot,
                )
                _bind_profile_and_runtime(context, profile, project_claim, runtime_snapshot, paths)
                sketch_result = sketch_module.validate_sketch(paths["sketch"])
                decision_result = decision_module.validate_decision(
                    paths["decision"],
                    project_root=frozen.root,
                    expected_implementation_profile=context.implementation_profile_id,
                )
                _bind_decision_artifact_refs(context, decision_result, paths["decision"])
                _bind_decision_metadata(context, profile, runtime_snapshot, decision_result, paths)
            except KernelWikiError:
                raise
            except Exception as error:
                _raise_authority_error(error)

            authority = AuthoritySnapshot(
                contract_version=context.contract_version,
                loop_contract_identity=_freeze_loop_identity(current_identity),
                profile=_deep_freeze(profile),
                project_claim=_deep_freeze(project_claim),
                sketch_result=_deep_freeze(sketch_result),
                decision_result=_deep_freeze(decision_result),
                artifact_hashes=_deep_freeze(frozen.artifact_hashes),
            )
            return _seal_authority_snapshot(authority)

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from kernelwiki_common import KernelWikiError, load_yaml_document, validate_root_relative_posix_path


LIFT_SCHEMA_KINDS = (
    "terminal_bundle",
    "experience_proposal",
    "experience_review",
    "historical_capture",
)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _fail(message: str, path: Path | None = None) -> None:
    raise KernelWikiError("lift-schema-invalid", message, path)


def _mapping(value: Any, label: str, path: Path | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{label} must be an object", path)
    return value


def _closed(value: Mapping[str, Any], fields: Any, label: str, path: Path | None) -> None:
    if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
        _fail(f"{label} field definition is invalid", path)
    if set(value) != set(fields):
        _fail(f"{label} must contain exactly its registered fields", path)


def _hash(value: Any, pattern: re.Pattern[str], label: str, path: Path | None) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(f"{label} has an invalid hash", path)


def load_lift_schema_registry(schema_path: Path) -> Mapping[str, Any]:
    path = Path(schema_path)
    registry = _mapping(load_yaml_document(path), "schema registry", path)
    definitions = _mapping(registry.get("field_definitions"), "field_definitions", path)
    required = {"loop_contract_identity", "bundle_artifact", *LIFT_SCHEMA_KINDS}
    if set(definitions) != required:
        _fail("field_definitions must contain the closed lift definitions", path)
    for kind in LIFT_SCHEMA_KINDS:
        if type(registry.get(kind)) is not int or registry[kind] != 1:
            _fail(f"{kind} schema version must be integer 1", path)
    for name, raw_definition in definitions.items():
        definition = _mapping(raw_definition, name, path)
        fields = definition.get("fields")
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or len(fields) != len(set(fields))
        ):
            _fail(f"{name}.fields must be unique nonempty strings", path)
    return registry


def _validate_identity(value: Any, definitions: Mapping[str, Any], path: Path | None) -> None:
    identity = _mapping(value, "loop_contract_identity", path)
    _closed(identity, definitions["loop_contract_identity"]["fields"], "loop_contract_identity", path)
    _hash(identity["repository_commit"], _HEX40, "repository_commit", path)
    _hash(identity["skill_tree_sha"], _HEX40, "skill_tree_sha", path)
    for label in ("validator_sha256", "schema_sha256"):
        hashes = _mapping(identity[label], label, path)
        for name, sha256 in hashes.items():
            if not isinstance(name, str) or not name:
                _fail(f"{label} names must be nonempty strings", path)
            _hash(sha256, _HEX64, f"{label}.{name}", path)


def _validate_bundle(
    document: Mapping[str, Any], definition: Mapping[str, Any], definitions: Mapping[str, Any], path: Path | None
) -> None:
    if document["contract_version"] not in definition["supported_contract_versions"]:
        _fail("contract_version is unsupported", path)
    _hash(document["terminal_commit"], _HEX40, "terminal_commit", path)
    _validate_identity(document["loop_contract_identity"], definitions, path)
    if document["measurement_exclusive"] is not False:
        _fail("measurement_exclusive must be false", path)

    artifacts = _mapping(document["artifacts"], "artifacts", path)
    required = set(definition["required_artifacts"])
    allowed = required | set(definition["optional_artifacts"])
    if not required.issubset(artifacts) or not set(artifacts).issubset(allowed):
        _fail("artifacts must contain every required artifact and no unknown artifact", path)
    artifact_fields = definitions["bundle_artifact"]["fields"]
    for name, raw_artifact in artifacts.items():
        artifact = _mapping(raw_artifact, f"artifact {name}", path)
        _closed(artifact, artifact_fields, f"artifact {name}", path)
        if artifact["name"] != name or type(artifact["required"]) is not bool:
            _fail(f"artifact {name} identity is invalid", path)
        if name in required and artifact["required"] is not True:
            _fail(f"artifact {name} must be required", path)
        _hash(artifact["sha256"], _HEX64, f"artifact {name} sha256", path)
        try:
            validate_root_relative_posix_path(artifact["path"])
        except ValueError as error:
            _fail(f"artifact {name} path is invalid: {error}", path)


def _walk_forbidden(value: Any, forbidden: set[str], path: Path | None) -> None:
    if isinstance(value, Mapping):
        found = forbidden & set(value)
        if found:
            _fail(f"proposal contains forbidden fields: {', '.join(sorted(found))}", path)
        for nested in value.values():
            _walk_forbidden(nested, forbidden, path)
    elif isinstance(value, list):
        for nested in value:
            _walk_forbidden(nested, forbidden, path)


def _validate_proposal(
    document: Mapping[str, Any], definition: Mapping[str, Any], definitions: Mapping[str, Any], path: Path | None
) -> None:
    _walk_forbidden(document, set(definition["forbidden_recursive_fields"]), path)
    if document["source_lane"] not in {"strict-current-vnext", "historical-manual"}:
        _fail("source_lane is invalid", path)
    if type(document["contract_version"]) is not int or document["contract_version"] < 1:
        _fail("contract_version must be a positive integer", path)
    if document["loop_contract_identity"] is not None:
        _validate_identity(document["loop_contract_identity"], definitions, path)
    for name, sha256 in _mapping(document["artifact_hashes"], "artifact_hashes", path).items():
        _hash(sha256, _HEX64, f"artifact_hashes.{name}", path)
    for label in ("terminal", "scope", "expected", "suggested_publication"):
        _mapping(document[label], label, path)
    if not isinstance(document["observed"], list):
        _fail("observed must be a list", path)
    for label in ("transfer_boundaries", "reconsider_when", "missing_evidence"):
        if not isinstance(document[label], list) or any(not isinstance(item, str) for item in document[label]):
            _fail(f"{label} must be a list of strings", path)


def _validate_review(document: Mapping[str, Any], path: Path | None) -> None:
    _hash(document["proposal_sha256"], _HEX64, "proposal_sha256", path)
    if document["decision"] not in {"include", "defer", "exclude"}:
        _fail("decision must be include, defer, or exclude", path)
    for label in ("proposal_id", "reviewed_by", "reviewed_at", "rationale"):
        if not isinstance(document[label], str) or not document[label].strip():
            _fail(f"{label} must be nonempty text", path)
    if document["publication_target"] is not None:
        _mapping(document["publication_target"], "publication_target", path)


def validate_lift_document(
    kind: str, document: Any, schema_path: Path, *, path: Path | None = None
) -> dict[str, Any]:
    registry = load_lift_schema_registry(schema_path)
    if kind not in LIFT_SCHEMA_KINDS:
        _fail(f"unknown lift schema kind: {kind}", path)
    value = _mapping(document, kind, path)
    definitions = registry["field_definitions"]
    definition = definitions[kind]
    _closed(value, definition["fields"], kind, path)
    if type(value["schema_version"]) is not int or value["schema_version"] != registry[kind]:
        _fail(f"{kind} schema_version must be integer {registry[kind]}", path)
    if kind == "terminal_bundle":
        _validate_bundle(value, definition, definitions, path)
    elif kind == "experience_proposal":
        _validate_proposal(value, definition, definitions, path)
    elif kind == "experience_review":
        _validate_review(value, path)
    return dict(value)

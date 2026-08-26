"""Machine-readable implementation profile and project capability claim validation.

A canonical implementation profile is language-neutral and structurally complete
even while individual capabilities are unknown. ``load_profile()`` resolves the
profile-local vendored schema copies, probe catalog, and approved archived
evidence inside the profile root so a frozen snapshot remains valid after the
canonical directory changes or disappears.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from vnext_common import (
    ContractValidationError,
    load_json_document,
    load_json_yaml_document,
    require_relative_artifact,
    sha256_canonical_json,
    sha256_file,
)


class ProfileValidationError(ContractValidationError):
    pass


CAPABILITY_STATUSES = frozenset({"supported", "constrained", "unknown", "unsupported", "prohibited"})
HINT_MODALITIES = frozenset({"required", "preferred", "exploratory"})
REQUIRED_SECTIONS = (
    "implementation_profile_id",
    "implementation_profile_version",
    "profile_status",
    "implementation",
    "profile_schema_ref",
    "profile_schema_sha256",
    "shared_profile_schema_ref",
    "shared_profile_schema_version",
    "shared_profile_schema_sha256",
    "identity_match",
    "runtime_launcher",
    "source_conformance",
    "capability_matrix",
    "probe_catalog",
    "fallback_and_unknown_policy",
    "profiler_evidence",
)
OPTIONAL_SECTIONS = frozenset({"resource_constraints", "configuration_constraints", "host_lifecycle"})
CONFIGURATION_FIELD_KINDS = frozenset({"launch-option", "compile-time-meta"})
FORBIDDEN_CLAIM_KEYS = frozenset(
    {"probe_result_ref", "probe_result_path", "result_ref", "result_path", "run_dir", "raw_result_ref"}
)


def _error(code: str, message: str, path: Path | None = None) -> ProfileValidationError:
    return ProfileValidationError(code, message, path)


def load_profile(path: Path) -> dict[str, Any]:
    path = Path(path)
    profile = load_json_yaml_document(path, artifact="implementation profile")
    _validate_required_sections(profile)
    _validate_identity(profile["identity_match"])
    _validate_implementation(profile["implementation"], profile["source_conformance"])
    _validate_capability_matrix(profile["capability_matrix"])
    _validate_configuration_constraints(profile.get("configuration_constraints", {}))
    _validate_probe_catalog(profile["probe_catalog"], path.parent)
    _validate_schema_copies(profile, path.parent)
    _validate_evidence_scopes(profile, path.parent)
    profile["_profile_path"] = path
    profile["_profile_sha256"] = sha256_file(path)
    return profile


def _validate_required_sections(profile: Mapping[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in profile:
            raise _error("profile-section-required", f"profile requires section {section!r}")
    status = profile.get("profile_status")
    if status not in {"partial", "complete"}:
        raise _error("profile-status-invalid", "profile_status must be partial or complete")
    for key in ("implementation_profile_id", "implementation_profile_version"):
        if key == "implementation_profile_id":
            value = profile[key]
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
                raise _error("profile-identity-invalid", "implementation_profile_id must be a safe identifier")
        elif isinstance(profile[key], bool) or not isinstance(profile[key], int) or profile[key] < 1:
            raise _error("profile-identity-invalid", "implementation_profile_version must be a positive integer")
    for key in ("profile_schema_ref", "shared_profile_schema_ref"):
        if not isinstance(profile[key], str) or not profile[key]:
            raise _error("profile-schema-ref-invalid", f"{key} must be a nonempty relative path")
    for key in ("profile_schema_sha256", "shared_profile_schema_sha256"):
        if not isinstance(profile[key], str) or not re.fullmatch(r"[0-9a-f]{64}", profile[key]):
            raise _error("profile-schema-hash-invalid", f"{key} must be a SHA-256 hex digest")
    shared_version = profile["shared_profile_schema_version"]
    if isinstance(shared_version, bool) or not isinstance(shared_version, int) or shared_version < 1:
        raise _error("profile-schema-version-invalid", "shared_profile_schema_version must be a positive integer")


def _validate_identity(identity_match: Mapping[str, Any]) -> None:
    if not isinstance(identity_match, dict):
        raise _error("profile-identity-invalid", "identity_match must be an object")
    for field in ("permitted_target_ids", "permitted_device_architectures", "match_rules"):
        value = identity_match.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise _error("profile-identity-invalid", f"identity_match.{field} must be a list of nonempty strings")


def _validate_implementation(implementation: Mapping[str, Any], source_conformance: Mapping[str, Any]) -> None:
    if not isinstance(implementation, dict):
        raise _error("profile-implementation-invalid", "implementation must be an object")
    for field in ("language", "backend", "runner_adapter"):
        if not isinstance(implementation.get(field), str) or not implementation[field].strip():
            raise _error("profile-implementation-invalid", f"implementation.{field} must be a nonempty string")
    if not isinstance(source_conformance, dict):
        raise _error("profile-source-conformance-invalid", "source_conformance must be an object")
    for field in ("analyzer", "binding_model"):
        if not isinstance(source_conformance.get(field), str) or not source_conformance[field].strip():
            raise _error("profile-source-conformance-invalid", f"source_conformance.{field} must be a nonempty string")


def _validate_capability_matrix(capability_matrix: Any) -> None:
    if not isinstance(capability_matrix, list) or not capability_matrix:
        raise _error("profile-capability-matrix-invalid", "capability_matrix must be a nonempty list")
    seen: set[str] = set()
    for entry in capability_matrix:
        if not isinstance(entry, dict):
            raise _error("profile-capability-invalid", "each capability entry must be an object")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise _error("profile-capability-invalid", "each capability entry requires a nonempty id")
        if identifier in seen:
            raise _error("profile-capability-duplicate", f"duplicate capability id {identifier!r}")
        seen.add(identifier)
        if entry.get("status") not in CAPABILITY_STATUSES:
            raise _error("profile-capability-status-invalid", f"capability {identifier!r} has an invalid status")
        for field in ("family", "capability_kind", "contract_name", "implementation_symbol"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise _error("profile-capability-invalid", f"capability {identifier!r} requires nonempty {field}")
        if not isinstance(entry.get("signature"), dict):
            raise _error("profile-capability-invalid", f"capability {identifier!r} requires a signature object")
        scope = entry.get("scope")
        if not isinstance(scope, dict):
            raise _error("profile-capability-invalid", f"capability {identifier!r} requires a scope object")
        for field in ("target_id", "runtime", "device_arch", "shape_signature"):
            if not isinstance(scope.get(field), str) or not scope[field].strip():
                raise _error("profile-capability-invalid", f"capability {identifier!r} scope requires nonempty {field}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list):
            raise _error("profile-capability-invalid", f"capability {identifier!r} evidence must be a list")
        for item in evidence:
            if not isinstance(item, dict):
                raise _error("profile-evidence-invalid", f"capability {identifier!r} evidence item must be an object")
            if item.get("review_status") != "approved":
                raise _error("profile-evidence-review-invalid", f"evidence {item.get('evidence_id')!r} must be review_status approved")
            for field in (
                "evidence_id",
                "archived_result_ref",
                "archived_result_sha256",
                "probe_id",
                "probe_definition_sha256",
                "result_sha256",
                "kind",
                "target_id",
                "toolchain_fingerprint",
                "device_arch",
                "runner_adapter",
                "launcher_context",
            ):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise _error("profile-evidence-invalid", f"evidence {item.get('evidence_id')!r} requires nonempty {field}")
            if item.get("provenance") not in {"observed", "inferred", "unknown"}:
                raise _error("profile-evidence-invalid", f"evidence {item.get('evidence_id')!r} provenance must be observed|inferred|unknown")
            for field in ("archived_result_sha256", "probe_definition_sha256", "result_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", item[field]):
                    raise _error("profile-evidence-hash-invalid", f"evidence {item.get('evidence_id')!r} {field} must be a SHA-256 hex digest")


def _validate_configuration_constraints(configuration_constraints: Any) -> None:
    if not configuration_constraints:
        return
    if not isinstance(configuration_constraints, dict):
        raise _error("profile-configuration-invalid", "configuration_constraints must be an object")
    fields = configuration_constraints.get("fields")
    if not isinstance(fields, list):
        raise _error("profile-configuration-invalid", "configuration_constraints.fields must be a list")
    names: set[str] = set()
    for field in fields:
        if not isinstance(field, dict):
            raise _error("profile-configuration-invalid", "each configuration field must be an object")
        name = field.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _error("profile-configuration-invalid", "each configuration field requires a nonempty name")
        if name in names:
            raise _error("profile-configuration-duplicate", f"duplicate configuration field {name!r}")
        names.add(name)
        if field.get("kind") not in CONFIGURATION_FIELD_KINDS:
            raise _error("profile-configuration-invalid", f"configuration field {name!r} kind must be launch-option|compile-time-meta")
        values = field.get("values")
        if not isinstance(values, list) or not values:
            raise _error("profile-configuration-invalid", f"configuration field {name!r} requires a nonempty finite values list")
        if any(isinstance(value, list) or isinstance(value, dict) for value in values):
            raise _error("profile-configuration-invalid", f"configuration field {name!r} values must be scalar")
        if len(set(json.dumps(value, sort_keys=True) for value in values)) != len(values):
            raise _error("profile-configuration-duplicate", f"configuration field {name!r} values must be unique")
        scope = field.get("scope")
        if not isinstance(scope, dict) or not scope:
            raise _error("profile-configuration-invalid", f"configuration field {name!r} requires a scope object")
    exclusions = configuration_constraints.get("cross_field_exclusions")
    if not isinstance(exclusions, list):
        raise _error("profile-configuration-invalid", "configuration_constraints.cross_field_exclusions must be a list")
    for exclusion in exclusions:
        if not isinstance(exclusion, list) or len(exclusion) < 2 or any(name not in names for name in exclusion):
            raise _error("profile-configuration-invalid", "each cross-field exclusion must name two or more declared fields")


def _validate_probe_catalog(probe_catalog: Any, profile_root: Path) -> None:
    if not isinstance(probe_catalog, list):
        raise _error("profile-probe-catalog-invalid", "probe_catalog must be a list")
    seen: set[str] = set()
    for entry in probe_catalog:
        if not isinstance(entry, dict):
            raise _error("profile-probe-catalog-invalid", "each probe catalog entry must be an object")
        probe_id = entry.get("probe_id")
        definition_path = entry.get("definition_path")
        definition_sha256 = entry.get("definition_sha256")
        if not isinstance(probe_id, str) or not probe_id.strip():
            raise _error("profile-probe-catalog-invalid", "each probe catalog entry requires a nonempty probe_id")
        if probe_id in seen:
            raise _error("profile-probe-catalog-duplicate", f"duplicate probe catalog id {probe_id!r}")
        seen.add(probe_id)
        if not isinstance(definition_path, str) or not definition_path:
            raise _error("profile-probe-catalog-invalid", f"probe {probe_id!r} requires a definition_path")
        if not isinstance(definition_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", definition_sha256):
            raise _error("profile-probe-catalog-invalid", f"probe {probe_id!r} definition_sha256 must be a SHA-256 hex digest")
        definition_file = require_relative_artifact(profile_root, definition_path)
        if sha256_file(definition_file) != definition_sha256:
            raise _error("profile-probe-hash-mismatch", f"probe {probe_id!r} definition hash does not match {definition_path}")


def _validate_schema_copies(profile: Mapping[str, Any], profile_root: Path) -> None:
    profile_schema = require_relative_artifact(profile_root, profile["profile_schema_ref"])
    shared_schema = require_relative_artifact(profile_root, profile["shared_profile_schema_ref"])
    if sha256_file(profile_schema) != profile["profile_schema_sha256"]:
        raise _error("profile-schema-hash-mismatch", "profile.schema.json hash does not match profile.yaml")
    if sha256_file(shared_schema) != profile["shared_profile_schema_sha256"]:
        raise _error("profile-schema-hash-mismatch", "shared-profile.schema.json hash does not match profile.yaml")
    if sha256_file(profile_schema) != sha256_file(shared_schema):
        raise _error("profile-schema-hash-mismatch", "local profile schema and shared schema copies must be byte-identical")
    try:
        load_json_document(shared_schema, artifact="shared profile schema")
    except ContractValidationError as error:
        raise _error("profile-schema-invalid", f"vendored schema is not a JSON object: {error.message}")


def _validate_evidence_scopes(profile: Mapping[str, Any], profile_root: Path) -> None:
    from validate_probe import validate_archived_evidence  # lazy: avoids circular import

    for entry in profile["capability_matrix"]:
        for item in entry.get("evidence") or []:
            archived = require_relative_artifact(profile_root, item["archived_result_ref"])
            archived_sha = sha256_file(archived)
            if archived_sha != item["archived_result_sha256"]:
                raise _error("profile-evidence-hash-mismatch", f"archived result {item['archived_result_ref']} hash mismatch")
            if item["archived_result_sha256"] != item["result_sha256"]:
                raise _error("profile-evidence-hash-mismatch", f"evidence {item['evidence_id']!r} archived hash must equal result hash")
            validate_archived_evidence(entry, item, profile, archived)


def _match_capability(
    capability_matrix: Sequence[Mapping[str, Any]],
    contract_name: str,
    signature: Mapping[str, Any],
) -> dict[str, Any] | None:
    for entry in capability_matrix:
        if entry["contract_name"] != contract_name:
            continue
        entry_signature = entry["signature"]
        if all(key in entry_signature and entry_signature[key] == value for key, value in signature.items()):
            return entry
    return None


def require_capability(
    profile: Mapping[str, Any],
    contract_name: str,
    signature: Mapping[str, Any],
    modality: str,
) -> dict[str, Any]:
    if modality not in HINT_MODALITIES:
        raise _error("profile-hint-modality-invalid", "modality must be required|preferred|exploratory")
    entry = _match_capability(profile["capability_matrix"], contract_name, signature)
    if modality == "required" and (entry is None or entry["status"] in {"unknown", "unsupported", "prohibited"}):
        raise _error("profile-required-capability-unproven", "unproven required capability")
    if entry is None:
        raise _error("profile-capability-unmatched", f"no capability matches contract {contract_name!r}")
    return entry


def validate_configuration_domain(
    profile: Mapping[str, Any],
    fields: Sequence[Mapping[str, Any]],
    scope: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return the finite deterministic legal configuration domain for *fields*.

    Every declared field and value must be covered by reviewed exact-scope
    profile legality; cross-field exclusions prune combinations; the declared
    order is preserved.
    """
    constraints = profile.get("configuration_constraints")
    if not isinstance(constraints, dict):
        raise _error("profile-legality-unavailable", "profile has no reviewed configuration legality")
    legal: dict[str, dict[str, Any]] = {field["name"]: field for field in constraints.get("fields", [])}
    if not isinstance(fields, list) or not fields:
        raise _error("profile-configuration-domain-invalid", "a configuration domain requires at least one field")

    declared: list[tuple[str, list[Any]]] = []
    for field in fields:
        if not isinstance(field, dict):
            raise _error("profile-configuration-domain-invalid", "each domain field must be an object")
        name = field.get("name")
        if not isinstance(name, str) or name not in legal:
            raise _error("profile-legality-unavailable", f"no reviewed legality for configuration field {name!r}")
        legality = legal[name]
        field_scope = legality.get("scope") or {}
        if not _scope_covers(field_scope, scope):
            raise _error("profile-legality-scope-mismatch", f"configuration field {name!r} legality does not cover the requested exact scope")
        values = field.get("values")
        if not isinstance(values, list) or not values:
            raise _error("profile-configuration-domain-invalid", f"configuration field {name!r} requires a finite value list")
        legal_values = legality["values"]
        legal_json = {json.dumps(value, sort_keys=True) for value in legal_values}
        normalized_values: list[Any] = []
        for value in values:
            if json.dumps(value, sort_keys=True) not in legal_json:
                status = _legality_status(profile, name, value)
                if status in {"unknown", "prohibited", "unsupported"}:
                    raise _error("profile-legality-unproven", f"configuration value {value!r} for {name!r} is {status}")
                raise _error("profile-legality-unavailable", f"configuration value {value!r} for {name!r} is not legal")
            if any(_same_value(existing, value) for existing in normalized_values):
                raise _error("profile-configuration-duplicate", f"duplicate configuration value {value!r} for {name!r}")
            normalized_values.append(value)
        declared.append((name, normalized_values))

    combinations: list[dict[str, Any]] = [{}]
    for name, values in declared:
        combinations = [
            {**combination, name: value}
            for combination in combinations
            for value in values
        ]

    exclusions = constraints.get("cross_field_exclusions") or []
    excluded_pairs = {tuple(sorted(pair)) for pair in exclusions}
    domain: list[dict[str, Any]] = []
    seen_configs: set[str] = set()
    for combination in combinations:
        names = sorted(combination)
        if any(tuple(sorted(names)) == pair or _exclusion_hit(combination, pair) for pair in excluded_pairs):
            continue
        key = json.dumps(combination, sort_keys=True)
        if key in seen_configs:
            raise _error("profile-configuration-duplicate", "duplicate normalized configuration in domain")
        seen_configs.add(key)
        domain.append(combination)
    return tuple(domain)


def _exclusion_hit(combination: Mapping[str, Any], pair: tuple[str, ...]) -> bool:
    return all(name in combination for name in pair)


def _same_value(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _legality_status(profile: Mapping[str, Any], name: str, value: Any) -> str:
    """Best-effort capability status for an illegal value; unknown by default."""
    for entry in profile.get("capability_matrix") or []:
        if entry.get("contract_name") == name or entry.get("id", "").endswith(name):
            return entry["status"]
    return "unknown"


def _scope_covers(field_scope: Mapping[str, Any], scope: Mapping[str, Any]) -> bool:
    return all(key in field_scope and field_scope[key] == value for key, value in scope.items())


def validate_project_claim(
    path: Path,
    *,
    profile: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the run-local claim against the loaded profile and run snapshot."""
    claim = load_json_document(Path(path), artifact="project capability claim")
    profile_path = Path(profile["_profile_path"])
    profile_sha = profile.get("_profile_sha256") or sha256_file(profile_path)

    _validate_identity_matching(claim, profile, profile_sha, snapshot)
    _validate_dispositions(claim.get("qualification_dispositions") or [])
    _validate_claim_contract(claim)
    return {"valid": True, "claim": claim}


def _validate_identity_matching(
    claim: Mapping[str, Any],
    profile: Mapping[str, Any],
    profile_sha: str,
    snapshot: Mapping[str, Any],
) -> None:
    if claim.get("implementation_profile_id") != profile["implementation_profile_id"]:
        raise _error("environment-blocked", "claim implementation_profile_id does not match the loaded profile")
    if claim.get("implementation_profile_version") != profile["implementation_profile_version"]:
        raise _error("environment-blocked", "claim implementation_profile_version does not match the loaded profile")
    if claim.get("implementation_profile_sha256") != profile_sha:
        raise _error("environment-blocked", "claim implementation_profile_sha256 does not match the loaded profile")
    target_id = claim.get("target_id")
    if not isinstance(target_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", target_id):
        raise _error("environment-blocked", "claim target_id must be a safe identifier")
    if target_id not in (profile["identity_match"].get("permitted_target_ids") or []):
        raise _error("environment-blocked", f"target_id {target_id!r} is not permitted by the profile identity match")
    if snapshot.get("target_id") != target_id:
        raise _error("environment-blocked", "run snapshot target_id does not match the claim")
    if snapshot.get("implementation_profile_id") != profile["implementation_profile_id"]:
        raise _error("environment-blocked", "run snapshot implementation_profile_id does not match the profile")
    device_arch = snapshot.get("device_arch")
    if device_arch not in (profile["identity_match"].get("permitted_device_architectures") or []):
        raise _error("environment-blocked", f"run snapshot device architecture {device_arch!r} does not match the profile")


def _validate_claim_contract(claim: Mapping[str, Any]) -> None:
    for field in ("primary_contract",):
        if not isinstance(claim.get(field), str) or not claim[field].strip():
            raise _error("claim-contract-invalid", f"claim requires nonempty {field}")
    if not isinstance(claim.get("primary_signature"), dict):
        raise _error("claim-contract-invalid", "claim requires a primary_signature object")
    if not isinstance(claim.get("runtime_fingerprint"), str) or not claim["runtime_fingerprint"].strip():
        raise _error("claim-contract-invalid", "claim requires a nonempty runtime_fingerprint")


def _validate_dispositions(dispositions: list[Any]) -> None:
    for disposition in dispositions:
        if not isinstance(disposition, dict):
            raise _error("claim-disposition-invalid", "each qualification disposition must be an object")
        allowed = {
            "disposition_id",
            "requirement",
            "requirement_sha256",
            "onboarding_outcome",
            "promotion_disposition",
            "fallback_authorized",
            "reason",
            "maintainer_confirmation",
            "probe_id",
            "probe_definition_sha256",
            "probe_result_sha256",
            "primary_remains_unknown",
        }
        unexpected = set(disposition) - allowed
        if any(key in disposition for key in FORBIDDEN_CLAIM_KEYS):
            raise _error("claim-raw-probe-ref", "disposition must not contain a raw probe-result reference")
        if unexpected:
            raise _error("claim-disposition-invalid", f"disposition contains undeclared fields {sorted(unexpected)!r}")
        requirement = disposition.get("requirement")
        if not isinstance(requirement, dict):
            raise _error("claim-disposition-invalid", "each disposition requires an embedded normalized requirement")
        for field in ("requirement_id", "primary_contract", "fallback_contract"):
            if not isinstance(requirement.get(field), str) or not requirement[field].strip():
                raise _error("claim-disposition-invalid", f"requirement {requirement.get('requirement_id')!r} requires nonempty {field}")
        if requirement.get("fallback_kind") not in {"semantic-accommodation", "algorithm-substitution"}:
            raise _error("claim-disposition-invalid", "requirement fallback_kind must be semantic-accommodation|algorithm-substitution")
        if requirement.get("probe_policy") not in {"optional", "before-fallback", "must-resolve"}:
            raise _error("claim-disposition-invalid", "requirement probe_policy must be optional|before-fallback|must-resolve")
        for field in ("primary_signature", "fallback_signature"):
            if not isinstance(requirement.get(field), dict):
                raise _error("claim-disposition-invalid", f"requirement {requirement.get('requirement_id')!r} requires {field} object")
        expected_hash = sha256_canonical_json(requirement)
        if disposition.get("requirement_sha256") != expected_hash:
            raise _error("claim-requirement-hash-mismatch", "embedded requirement hash does not match canonical JSON hash")

        confirmation = disposition.get("maintainer_confirmation")
        if confirmation is not None:
            if not isinstance(confirmation, dict):
                raise _error("claim-disposition-invalid", "maintainer_confirmation must be an object")
            for field in ("confirmed_by", "confirmed_at"):
                if not isinstance(confirmation.get(field), str) or not confirmation[field].strip():
                    raise _error("claim-disposition-invalid", f"maintainer_confirmation requires nonempty {field}")
            if confirmation.get("method") not in {"explicit-user-instruction", "maintainer-reviewed-commit"}:
                raise _error("claim-disposition-invalid", "maintainer_confirmation method must be explicit-user-instruction|maintainer-reviewed-commit")
            confirmed_at = confirmation["confirmed_at"]
            if not re.search(r"Z$|[+-][0-9]{2}:[0-9]{2}$", confirmed_at):
                raise _error("claim-disposition-invalid", "maintainer_confirmation confirmed_at must be UTC RFC 3339")

        fallback_authorized = disposition.get("fallback_authorized")
        if not isinstance(fallback_authorized, bool):
            raise _error("claim-disposition-invalid", "fallback_authorized must be a boolean")
        if fallback_authorized:
            if confirmation is None:
                raise _error("claim-fallback-unconfirmed", "fallback authorization requires maintainer confirmation")
            if disposition.get("primary_remains_unknown") is not True:
                raise _error("claim-disposition-invalid", "authorized fallback requires primary_remains_unknown true")
            if not isinstance(disposition.get("reason"), str) or not disposition["reason"].strip():
                raise _error("claim-disposition-invalid", "authorized fallback requires a nonempty reason")
            if disposition.get("promotion_disposition") not in {"declined", "deferred", "not-applicable"}:
                raise _error("claim-disposition-invalid", "authorized fallback requires promotion_disposition declined|deferred|not-applicable")
        if requirement["fallback_kind"] == "algorithm-substitution" and not fallback_authorized:
            raise _error("claim-silent-substitution", "algorithm substitution without explicit fallback authorization is invalid")
        if requirement["probe_policy"] == "must-resolve" and fallback_authorized:
            raise _error("claim-disposition-invalid", "must-resolve never authorizes a fallback")
        if requirement["probe_policy"] == "before-fallback" and not fallback_authorized:
            raise _error("claim-fallback-unresolved", "before-fallback requirement remains unresolved without authorization")
        if requirement["probe_policy"] == "optional" and fallback_authorized:
            raise _error("claim-disposition-invalid", "an optional probe policy does not require or allow fallback authorization")

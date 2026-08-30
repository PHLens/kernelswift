"""Probe definition validation, run validation, and pure demand-scoped selection.

One versioned probe contract serves both pre-campaign profile onboarding and
bounded campaign-local capability checks while keeping their ownership and
authority distinct. ``select_profile_probes()`` is a pure operation: it never
mutates the filesystem and never launches a subprocess.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from vnext_common import (
    ContractValidationError,
    create_exclusive_directory,
    load_json_document,
    require_relative_artifact,
    sha256_canonical_json,
    sha256_file,
)


class ProbeValidationError(ContractValidationError):
    pass


RUN_SUMMARIES = frozenset({"evidence-ready", "partial", "environment-blocked", "probe-failed"})
RESULT_LEVELS = frozenset({"observed", "inferred", "unknown"})
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ALLOWED_PLACEHOLDERS = frozenset(
    {
        "{interpreter}",
        "{probe_inputs_root}",
        "{probe_run_dir}",
        "{result_payload_path}",
        "{runtime_snapshot_path}",
        "{target_id}",
    }
)
REQUIRED_RUNTIME_FIELDS = frozenset(
    {"interpreter", "device", "toolchain", "device_arch", "runner_adapter", "bootstrap_modules", "synchronize_api"}
)
SECRET_KEYS = ("secret", "token", "password")


def _error(code: str, message: str, path: Path | None = None) -> ProbeValidationError:
    return ProbeValidationError(code, message, path)


@dataclass(frozen=True)
class ProbeSelection:
    probe_id: str
    requirement_id: str
    definition_path: str


@dataclass(frozen=True)
class RequirementDisposition:
    requirement_id: str
    outcome: str


@dataclass(frozen=True)
class QualificationPlan:
    selections: tuple[ProbeSelection, ...]
    dispositions: tuple[RequirementDisposition, ...]


def validate_probe_definition(path: Path, *, profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one versioned probe definition against the loaded profile."""
    definition = load_json_document(Path(path), artifact="probe definition")
    if definition.get("schema_version") != 1:
        raise _error("probe-schema-version", "probe definition schema_version must be 1", Path(path))
    probe_id = definition.get("probe_id")
    if not isinstance(probe_id, str) or not SAFE_ID_PATTERN.fullmatch(probe_id):
        raise _error("probe-id-invalid", "probe_id must match [A-Za-z0-9._-]+", Path(path))
    if definition.get("implementation_profile_id") != profile["implementation_profile_id"]:
        raise _error("probe-profile-mismatch", "probe definition implementation_profile_id does not match the profile", Path(path))
    for field in ("family", "purpose", "scope_kind"):
        if not isinstance(definition.get(field), str) or not definition[field].strip():
            raise _error("probe-field-required", f"probe {probe_id!r} requires nonempty {field}", Path(path))
    capability_ids = definition.get("capability_ids")
    if not isinstance(capability_ids, list) or not capability_ids:
        raise _error("probe-capability-ids-required", f"probe {probe_id!r} requires at least one capability id", Path(path))
    profile_ids = {entry["id"] for entry in profile["capability_matrix"]}
    for capability_id in capability_ids:
        if not isinstance(capability_id, str) or capability_id not in profile_ids:
            raise _error("probe-capability-unknown", f"probe {probe_id!r} declares unknown capability {capability_id!r}", Path(path))
    scope_template = definition.get("scope_template")
    if not isinstance(scope_template, dict) or not scope_template:
        raise _error("probe-scope-required", f"probe {probe_id!r} requires a scope_template object", Path(path))

    input_artifacts = definition.get("input_artifacts")
    if not isinstance(input_artifacts, list):
        raise _error("probe-input-artifacts-invalid", f"probe {probe_id!r} input_artifacts must be a list", Path(path))
    profile_root = Path(profile["_profile_path"]).parent
    seen_run_paths: set[str] = set()
    for artifact in input_artifacts:
        if not isinstance(artifact, dict):
            raise _error("probe-input-artifact-invalid", f"probe {probe_id!r} input artifact must be an object", Path(path))
        relative = artifact.get("path")
        recorded_sha = artifact.get("sha256")
        run_path = artifact.get("run_path")
        if not isinstance(relative, str) or not relative:
            raise _error("probe-input-artifact-invalid", f"probe {probe_id!r} input artifact requires a path", Path(path))
        if not isinstance(recorded_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_sha):
            raise _error("probe-input-artifact-invalid", f"probe {probe_id!r} input artifact requires a SHA-256 hash", Path(path))
        if not isinstance(run_path, str) or not run_path or "/" in run_path or "\\" in run_path or run_path in {"", ".", ".."}:
            raise _error("probe-input-artifact-invalid", f"probe {probe_id!r} input artifact run_path must be a safe filename", Path(path))
        if run_path in seen_run_paths:
            raise _error("probe-input-artifact-invalid", f"probe {probe_id!r} duplicate run_path {run_path!r}", Path(path))
        seen_run_paths.add(run_path)
        artifact_file = require_relative_artifact(profile_root, relative)
        if sha256_file(artifact_file) != recorded_sha:
            raise _error("probe-input-hash-mismatch", f"probe {probe_id!r} input artifact {relative!r} hash mismatch", Path(path))

    runner = definition.get("runner")
    if not isinstance(runner, dict):
        raise _error("probe-runner-invalid", f"probe {probe_id!r} requires a runner object", Path(path))
    if runner.get("kind") != "command":
        raise _error("probe-runner-kind-invalid", f"probe {probe_id!r} runner kind must be command", Path(path))
    argv = runner.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
        raise _error("probe-runner-argv-invalid", f"probe {probe_id!r} runner argv must be a nonempty string array", Path(path))
    timeout = runner.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise _error("probe-runner-timeout-invalid", f"probe {probe_id!r} runner timeout_seconds must be positive", Path(path))

    required_runtime_fields = definition.get("required_runtime_fields")
    if not isinstance(required_runtime_fields, list) or not required_runtime_fields:
        raise _error("probe-runtime-fields-required", f"probe {probe_id!r} requires required_runtime_fields", Path(path))
    unknown_fields = set(required_runtime_fields) - REQUIRED_RUNTIME_FIELDS
    if unknown_fields:
        raise _error("probe-runtime-field-unknown", f"probe {probe_id!r} declares unknown runtime fields {sorted(unknown_fields)!r}", Path(path))

    allowlisted = ALLOWED_PLACEHOLDERS | {f"{{{field}}}" for field in required_runtime_fields}
    for argument in argv:
        for match in re.finditer(r"\{[A-Za-z0-9_.-]+\}", argument):
            if match.group(0) not in allowlisted:
                raise _error("probe-placeholder-invalid", f"probe {probe_id!r} argv uses disallowed placeholder {match.group(0)}", Path(path))

    return {"valid": True, "definition": definition}


def validate_probe_run(run_dir: Path) -> dict[str, Any]:
    """Validate a completed run directory independently of the current canonical profile."""
    run_dir = Path(run_dir)
    run_path = run_dir / "run.json"
    run = load_json_document(run_path, artifact="probe run")
    if run.get("schema_version") != 1:
        raise _error("probe-run-schema-version", "probe run schema_version must be 1", run_path)
    summary = run.get("summary")
    if summary not in RUN_SUMMARIES:
        raise _error("probe-run-summary-invalid", "probe run summary must be evidence-ready|partial|environment-blocked|probe-failed", run_path)
    for field in ("run_id", "target_id", "implementation_profile_id"):
        if not isinstance(run.get(field), str) or not run[field]:
            raise _error("probe-run-field-invalid", f"probe run requires nonempty {field}", run_path)
    if not isinstance(run.get("implementation_profile_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", run["implementation_profile_sha256"]):
        raise _error("probe-run-field-invalid", "probe run requires implementation_profile_sha256", run_path)

    _validate_frozen_inputs(run_dir, run)
    requested_ids = run.get("requested_probe_ids")
    if not isinstance(requested_ids, list) or not requested_ids:
        raise _error("probe-run-field-invalid", "probe run requires requested_probe_ids", run_path)
    for probe_id in requested_ids:
        result_path = run_dir / "results" / f"{probe_id}.json"
        if not result_path.is_file():
            raise _error("probe-result-missing", f"probe run is missing result {probe_id}.json", result_path)
        _validate_result_file(result_path, run, run_dir, summary)
    return {"valid": True, "run": run, "summary": summary, "run_dir": run_dir}


def _validate_frozen_inputs(run_dir: Path, run: Mapping[str, Any]) -> None:
    inputs = run.get("inputs")
    if not isinstance(inputs, dict):
        raise _error("probe-run-inputs-invalid", "probe run inputs must be an object")
    for relative, record in inputs.items():
        if not isinstance(record, dict):
            raise _error("probe-run-inputs-invalid", f"probe run input {relative!r} record must be an object")
        actual = run_dir / relative
        if not actual.is_file():
            raise _error("probe-run-input-missing", f"frozen input {relative!r} is missing")
        byte_count = actual.stat().st_size
        if byte_count != record.get("byte_count") or sha256_file(actual) != record.get("sha256"):
            raise _error("probe-run-input-hash-mismatch", f"frozen input {relative!r} hash or byte count mismatch")


def _validate_result_file(result_path: Path, run: Mapping[str, Any], run_dir: Path, summary: str) -> None:
    result = load_json_document(result_path, artifact="probe result")
    if result.get("schema_version") != 1:
        raise _error("probe-result-schema-version", "probe result schema_version must be 1", result_path)
    if result.get("probe_id") != result_path.stem:
        raise _error("probe-result-id-mismatch", "probe result probe_id must match its filename", result_path)
    if result.get("implementation_profile_id") != run["implementation_profile_id"]:
        raise _error("probe-result-profile-mismatch", "probe result implementation_profile_id must match the run", result_path)
    if result.get("target_id") != run["target_id"]:
        raise _error("probe-result-target-mismatch", "probe result target_id must match the run", result_path)
    if not isinstance(result.get("definition_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", result["definition_sha256"]):
        raise _error("probe-result-field-invalid", "probe result requires definition_sha256", result_path)
    observations = result.get("observations")
    if not isinstance(observations, list):
        raise _error("probe-result-observations-invalid", "probe result observations must be a list", result_path)
    seen: set[str] = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise _error("probe-result-observations-invalid", "each observation must be an object", result_path)
        capability_id = observation.get("capability_id")
        if not isinstance(capability_id, str) or capability_id in seen:
            raise _error("probe-result-observations-invalid", "observations require unique capability ids", result_path)
        seen.add(capability_id)
        if observation.get("level") not in RESULT_LEVELS:
            raise _error("probe-result-observations-invalid", "observation level must be observed|inferred|unknown", result_path)
        if not isinstance(observation.get("numerically_checked"), bool):
            raise _error("probe-result-observations-invalid", "observation requires a numerically_checked boolean", result_path)
    _validate_summary_consistency(result, observations, summary, result_path, run_dir)
    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        raise _error("probe-result-evidence-invalid", "probe result evidence must be a list", result_path)
    for item in evidence:
        if not isinstance(item, dict):
            raise _error("probe-result-evidence-invalid", "each evidence record must be an object", result_path)
        relative = item.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise _error("probe-result-evidence-invalid", "each evidence record requires a relative_path", result_path)
        actual = run_dir / relative
        if not actual.is_file():
            raise _error("probe-result-evidence-missing", f"evidence {relative!r} is missing", result_path)
        if actual.stat().st_size != item.get("byte_count") or sha256_file(actual) != item.get("sha256"):
            raise _error("probe-result-evidence-hash-mismatch", f"evidence {relative!r} hash or byte count mismatch", result_path)


def _validate_summary_consistency(
    result: Mapping[str, Any],
    observations: list[Any],
    summary: str,
    result_path: Path,
    run_dir: Path,
) -> None:
    checked = [o for o in observations if o.get("level") == "observed" and o.get("numerically_checked") is True]
    if summary == "evidence-ready":
        if not observations or len(checked) != len(observations):
            raise _error("probe-result-summary-inconsistent", "evidence-ready requires every observation observed and numerically checked", result_path)
        definition_path = run_dir / "inputs" / "probe-definition.json"
        definition = load_json_document(definition_path, artifact="probe definition")
        declared = set(definition.get("capability_ids") or [])
        observed_ids = {o.get("capability_id") for o in observations}
        if not declared.issubset(observed_ids):
            raise _error("probe-result-summary-inconsistent", "evidence-ready requires coverage of every declared capability id", result_path)
    elif summary == "partial":
        if not observations or len(checked) == len(observations):
            raise _error("probe-result-summary-inconsistent", "partial requires at least one unobserved or unchecked observation", result_path)
    elif summary == "probe-failed":
        if not (result.get("timed_out") is True or result.get("exit_code") != 0) and observations:
            raise _error("probe-result-summary-inconsistent", "probe-failed requires a timeout, nonzero exit, or absent observations", result_path)
    elif summary == "environment-blocked":
        if observations:
            raise _error("probe-result-summary-inconsistent", "environment-blocked runs carry no observations", result_path)


def _match_capability_entry(
    profile: Mapping[str, Any],
    contract_name: str,
    signature: Mapping[str, Any],
) -> dict[str, Any] | None:
    for entry in profile["capability_matrix"]:
        if entry["contract_name"] != contract_name:
            continue
        entry_signature = entry["signature"]
        if all(key in entry_signature and entry_signature[key] == value for key, value in signature.items()):
            return entry
    return None


def select_profile_probes(
    profile: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    runtime_snapshot: Mapping[str, Any],
) -> QualificationPlan:
    """Pure demand-scoped probe selection over explicit requirements only."""
    if not isinstance(requirements, list):
        raise _error("probe-requirements-invalid", "requirements must be a list")
    target_id = runtime_snapshot.get("target_id")
    if target_id not in (profile["identity_match"].get("permitted_target_ids") or []):
        raise _error("environment-blocked", f"runtime target_id {target_id!r} is not permitted by the profile")
    device_arch = runtime_snapshot.get("device_arch")
    if device_arch not in (profile["identity_match"].get("permitted_device_architectures") or []):
        raise _error("environment-blocked", f"runtime device architecture {device_arch!r} is not permitted by the profile")

    catalog = {entry["probe_id"]: entry for entry in profile.get("probe_catalog") or []}
    definitions: dict[str, dict[str, Any]] = {}
    profile_root = Path(profile["_profile_path"]).parent
    for probe_id, entry in catalog.items():
        definition_path = entry.get("definition_path")
        definition_file = require_relative_artifact(profile_root, definition_path)
        if sha256_file(definition_file) != entry.get("definition_sha256"):
            raise _error("probe-catalog-hash-mismatch", f"catalog definition {probe_id!r} hash mismatch")
        definition = load_json_document(definition_file, artifact="probe definition")
        if definition.get("probe_id") != probe_id:
            raise _error("probe-catalog-id-mismatch", f"catalog entry {probe_id!r} definition probe_id mismatch")
        definitions[probe_id] = definition

    fallback_policies = {
        (policy.get("primary_contract"), json.dumps(policy.get("primary_signature") or {}, sort_keys=True)): policy
        for policy in (profile.get("fallback_and_unknown_policy") or {}).get("probe_policies") or []
    }

    selections: list[ProbeSelection] = []
    dispositions: list[RequirementDisposition] = []
    for requirement in sorted(requirements, key=lambda item: item.get("requirement_id", "")):
        if not isinstance(requirement, dict):
            raise _error("probe-requirement-invalid", "each requirement must be an object")
        requirement_id = requirement.get("requirement_id")
        if not isinstance(requirement_id, str) or not requirement_id:
            raise _error("probe-requirement-invalid", "each requirement requires a nonempty requirement_id")
        primary_contract = requirement.get("primary_contract")
        primary_signature = requirement.get("primary_signature")
        fallback_contract = requirement.get("fallback_contract")
        fallback_signature = requirement.get("fallback_signature")
        fallback_kind = requirement.get("fallback_kind")
        probe_policy = requirement.get("probe_policy")
        if not isinstance(primary_contract, str) or not isinstance(primary_signature, dict):
            raise _error("probe-requirement-invalid", f"requirement {requirement_id!r} requires primary_contract and primary_signature")
        if probe_policy not in {"optional", "before-fallback", "must-resolve"}:
            raise _error("probe-requirement-invalid", f"requirement {requirement_id!r} probe_policy must be optional|before-fallback|must-resolve")
        if probe_policy == "optional":
            dispositions.append(RequirementDisposition(requirement_id=requirement_id, outcome="optional-not-selected"))
            continue

        primary_entry = _match_capability_entry(profile, primary_contract, primary_signature)
        if primary_entry is None:
            dispositions.append(RequirementDisposition(requirement_id=requirement_id, outcome="no-exact-probe"))
            continue
        if primary_entry["status"] != "unknown":
            dispositions.append(RequirementDisposition(requirement_id=requirement_id, outcome="already-resolved"))
            continue

        if probe_policy == "before-fallback":
            if not isinstance(fallback_contract, str) or not isinstance(fallback_signature, dict) or fallback_kind != "algorithm-substitution":
                raise _error("probe-requirement-invalid", f"requirement {requirement_id!r} before-fallback requires an algorithm-substitution fallback")

        matches = [
            probe_id
            for probe_id, definition in definitions.items()
            if _definition_covers(definition, primary_entry, primary_signature)
        ]
        if not matches:
            dispositions.append(RequirementDisposition(requirement_id=requirement_id, outcome="no-exact-probe"))
            continue
        if len(matches) > 1:
            raise _error("ambiguous-profile-probe-selection", f"multiple exact probes match requirement {requirement_id!r}: {sorted(matches)}")

        if probe_policy == "before-fallback":
            fallback_entry = _match_capability_entry(profile, fallback_contract, fallback_signature)
            policy_key = (primary_contract, json.dumps(primary_signature, sort_keys=True))
            if fallback_entry is None or fallback_entry["status"] not in {"supported", "constrained"} or policy_key not in fallback_policies:
                dispositions.append(RequirementDisposition(requirement_id=requirement_id, outcome="no-fallback"))
                continue

        probe_id = matches[0]
        selections.append(
            ProbeSelection(
                probe_id=probe_id,
                requirement_id=requirement_id,
                definition_path=catalog[probe_id]["definition_path"],
            )
        )
        dispositions.append(RequirementDisposition(requirement_id=requirement_id, outcome="selected"))

    selections.sort(key=lambda item: (item.requirement_id, item.probe_id))
    return QualificationPlan(selections=tuple(selections), dispositions=tuple(dispositions))


def _definition_covers(
    definition: Mapping[str, Any],
    primary_entry: Mapping[str, Any],
    primary_signature: Mapping[str, Any],
) -> bool:
    if primary_entry["id"] not in (definition.get("capability_ids") or []):
        return False
    scope_template = definition.get("scope_template")
    if not isinstance(scope_template, dict):
        return False
    return all(key in scope_template and scope_template[key] == value for key, value in primary_signature.items())


def validate_archived_evidence(
    entry: Mapping[str, Any],
    item: Mapping[str, Any],
    profile: Mapping[str, Any],
    archived: Path,
) -> None:
    """Validate an approved archived evidence record as a probe-result payload."""
    payload = load_json_document(archived, artifact="archived probe result")
    if payload.get("schema_version") != 1:
        raise _error("probe-result-schema-version", "archived probe result schema_version must be 1", archived)
    if payload.get("probe_id") != item["probe_id"]:
        raise _error("probe-result-id-mismatch", "archived probe result probe_id must match the evidence record", archived)
    if payload.get("implementation_profile_id") != profile["implementation_profile_id"]:
        raise _error("probe-result-profile-mismatch", "archived probe result implementation_profile_id must match the profile", archived)
    if payload.get("target_id") != item["target_id"]:
        raise _error("probe-result-target-mismatch", "archived probe result target_id must match the evidence record", archived)
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise _error("probe-result-observations-invalid", "archived probe result requires observations", archived)
    observed: dict[str, dict[str, Any]] = {}
    for observation in observations:
        if not isinstance(observation, dict) or not isinstance(observation.get("capability_id"), str):
            raise _error("probe-result-observations-invalid", "archived observation requires a capability_id", archived)
        observed[observation["capability_id"]] = observation
    if entry["id"] not in observed:
        raise _error("probe-result-observations-invalid", "approved evidence must observe its own capability id", archived)
    if item.get("provenance") == "observed":
        observation = observed[entry["id"]]
        if observation.get("level") != "observed" or observation.get("numerically_checked") is not True:
            raise _error("probe-result-observations-invalid", "approved observed evidence must be numerically checked", archived)
    observed_scope = payload.get("observed_scope")
    if isinstance(observed_scope, dict):
        for key, value in observed_scope.items():
            approved = entry["scope"].get(key)
            if approved not in (None, "", "unregistered") and approved != value:
                raise _error("profile-evidence-scope-wider", f"approved scope for {entry['id']!r} is broader than the observation scope", archived)

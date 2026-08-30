#!/usr/bin/env python3
"""Shell-free bounded profile-probe command runner.

Creates an isolated run-local probe directory under
``<output-root>/probes/<target-id>/<run-id>/``, freezes every validated input,
executes the declarative argv without a shell under a bounded timeout, and
writes normalized atomic artifacts. It never mutates the canonical profile and
never allocates campaign state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from vnext_common import (
    ContractValidationError,
    create_exclusive_directory,
    load_json_document,
    require_relative_artifact,
    sha256_canonical_json,
    sha256_file,
    write_json_atomic,
)
from validate_profile import ProfileValidationError, load_profile
from validate_probe import (
    ProbeValidationError,
    SAFE_ID_PATTERN,
    SECRET_KEYS,
    validate_probe_definition,
)


def _error(code: str, message: str, path: Path | None = None) -> ProbeValidationError:
    return ProbeValidationError(code, message, path)


def _record(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {"relative_path": relative, "byte_count": path.stat().st_size, "sha256": sha256_file(path)}


def run_profile_probe(
    *,
    profile_path: Path,
    probe_id: str,
    target_id: str,
    runtime_snapshot_path: Path,
    output_root: Path,
    qualification_requirement_path: Path | None = None,
    run_id: str | None = None,
) -> Path:
    profile_path = Path(profile_path).resolve()
    runtime_snapshot_path = Path(runtime_snapshot_path).resolve()
    output_root = Path(output_root).resolve()

    if not profile_path.is_absolute():
        raise _error("probe-path-absolute", "profile path must be absolute")
    if not runtime_snapshot_path.is_absolute():
        raise _error("probe-path-absolute", "runtime snapshot path must be absolute")
    if not isinstance(probe_id, str) or not SAFE_ID_PATTERN.fullmatch(probe_id):
        raise _error("probe-id-invalid", "probe_id must match [A-Za-z0-9._-]+")
    if not isinstance(target_id, str) or not SAFE_ID_PATTERN.fullmatch(target_id):
        raise _error("probe-target-id-invalid", "target_id must match [A-Za-z0-9._-]+")
    if run_id is not None and (not isinstance(run_id, str) or not SAFE_ID_PATTERN.fullmatch(run_id)):
        raise _error("probe-run-id-invalid", "run_id must match [A-Za-z0-9._-]+")

    profile = load_profile(profile_path)
    catalog = {entry["probe_id"]: entry for entry in profile.get("probe_catalog") or []}
    if probe_id not in catalog:
        raise _error("probe-catalog-missing", f"probe {probe_id!r} is not in the profile probe catalog")
    catalog_entry = catalog[probe_id]
    definition_file = require_relative_artifact(profile_path.parent, catalog_entry["definition_path"])
    if sha256_file(definition_file) != catalog_entry["definition_sha256"]:
        raise _error("probe-catalog-hash-mismatch", f"catalog definition {probe_id!r} hash mismatch")
    definition = validate_probe_definition(definition_file, profile=profile)["definition"]

    runtime_snapshot = load_json_document(runtime_snapshot_path, artifact="runtime snapshot")
    for key in SECRET_KEYS:
        if any(key in field.lower() for field in runtime_snapshot):
            raise _error("probe-runtime-secret", "runtime snapshot must not carry secret environment values")
    required_runtime_fields = definition["required_runtime_fields"]
    missing = [field for field in required_runtime_fields if field not in runtime_snapshot]
    if missing:
        raise _error("probe-runtime-fields-missing", f"runtime snapshot is missing required fields {missing}")
    interpreter = runtime_snapshot["interpreter"]
    if not isinstance(interpreter, str) or not interpreter:
        raise _error("probe-interpreter-invalid", "runtime snapshot interpreter must be a nonempty string")

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = create_exclusive_directory(output_root / "probes" / target_id / run_id)
    inputs_root = run_dir / "inputs"
    payload_root = inputs_root / "payload"
    results_root = run_dir / "results"
    evidence_root = run_dir / "evidence"
    results_root.mkdir()
    evidence_root.mkdir()

    environment_blocked_reason = _prelaunch_block(profile, target_id, runtime_snapshot, interpreter)
    if environment_blocked_reason is not None:
        return _write_blocked_run(
            run_dir=run_dir,
            profile=profile,
            definition_file=definition_file,
            target_id=target_id,
            runtime_snapshot=runtime_snapshot,
            definition=definition,
            definition_sha256=catalog_entry["definition_sha256"],
            run_id=run_id,
            reason=environment_blocked_reason,
        )

    inputs: dict[str, dict[str, Any]] = {}
    _freeze_into(inputs, inputs_root, "profile.snapshot.yaml", profile_path)
    _freeze_into(inputs, inputs_root, "probe-definition.json", definition_file)
    _freeze_into(inputs, inputs_root, "runtime-snapshot.json", runtime_snapshot_path)
    if qualification_requirement_path is not None:
        qualification_requirement_path = Path(qualification_requirement_path).resolve()
        if not qualification_requirement_path.is_absolute():
            raise _error("probe-path-absolute", "qualification requirement path must be absolute")
        load_json_document(qualification_requirement_path, artifact="qualification requirement")
        _freeze_into(inputs, inputs_root, "qualification-requirement.json", qualification_requirement_path)
    for artifact in definition.get("input_artifacts") or []:
        source = require_relative_artifact(profile_path.parent, artifact["path"])
        _freeze_into(inputs, inputs_root, f"payload/{artifact['run_path']}", source)

    result_payload_path = evidence_root / f"{probe_id}.payload.json"
    argv = _resolve_argv(
        definition["runner"]["argv"],
        definition=definition,
        interpreter=interpreter,
        probe_inputs_root=payload_root,
        probe_run_dir=run_dir,
        result_payload_path=result_payload_path,
        runtime_snapshot_path=inputs_root / "runtime-snapshot.json",
        target_id=target_id,
        runtime_snapshot=runtime_snapshot,
    )
    cwd = run_dir
    started = datetime.now(timezone.utc)
    timed_out = False
    exit_code = None
    stdout_bytes = b""
    stderr_bytes = b""
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            timeout=float(definition["runner"]["timeout_seconds"]),
            cwd=str(cwd),
            check=False,
        )
        exit_code = completed.returncode
        stdout_bytes = completed.stdout
        stderr_bytes = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout_bytes = error.stdout or b""
        stderr_bytes = error.stderr or b""
    except OSError as error:
        raise _error("environment-blocked", f"cannot start probe payload: {error}", run_dir) from error
    ended = datetime.now(timezone.utc)

    stdout_path = evidence_root / f"{probe_id}.stdout.log"
    stderr_path = evidence_root / f"{probe_id}.stderr.log"
    stdout_path.write_bytes(stdout_bytes)
    stderr_path.write_bytes(stderr_bytes)

    evidence_records = [_record(stdout_path, run_dir), _record(stderr_path, run_dir)]

    payload: dict[str, Any] | None = None
    if result_payload_path.is_file() and not timed_out and exit_code == 0:
        try:
            payload = load_json_document(result_payload_path, artifact="probe payload")
            evidence_records.append(_record(result_payload_path, run_dir))
        except ContractValidationError:
            payload = None

    result = _build_result(
        definition=definition,
        definition_sha256=catalog_entry["definition_sha256"],
        profile=profile,
        target_id=target_id,
        runtime_snapshot=runtime_snapshot,
        argv=argv,
        exit_code=exit_code,
        timed_out=timed_out,
        payload=payload,
        evidence=evidence_records,
    )
    summary = _classify(definition, exit_code, timed_out, payload)
    result["summary"] = summary
    write_json_atomic(results_root / f"{probe_id}.json", result)

    run_document = {
        "schema_version": 1,
        "run_id": run_id,
        "target_id": target_id,
        "implementation_profile_id": profile["implementation_profile_id"],
        "implementation_profile_path": profile_path.as_posix(),
        "implementation_profile_version": profile["implementation_profile_version"],
        "implementation_profile_sha256": profile["_profile_sha256"],
        "runtime_fingerprint": sha256_canonical_json(runtime_snapshot),
        "requested_probe_ids": [probe_id],
        "definition_sha256": catalog_entry["definition_sha256"],
        "start_timestamp": started.isoformat(),
        "end_timestamp": ended.isoformat(),
        "summary": summary,
        "inputs": inputs,
        "results": [_record(results_root / f"{probe_id}.json", run_dir)],
        "evidence": evidence_records,
    }
    if qualification_requirement_path is not None:
        run_document["qualification_requirement_sha256"] = sha256_file(Path(qualification_requirement_path))
    write_json_atomic(run_dir / "run.json", run_document)
    return run_dir


def _freeze_into(inputs: dict[str, dict[str, Any]], inputs_root: Path, relative: str, source: Path) -> None:
    target = inputs_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    run_relative = f"inputs/{relative}"
    inputs[run_relative] = {"relative_path": run_relative, "byte_count": target.stat().st_size, "sha256": sha256_file(target)}


def _prelaunch_block(
    profile: Mapping[str, Any],
    target_id: str,
    runtime_snapshot: Mapping[str, Any],
    interpreter: str,
) -> str | None:
    """Return a stable environment-blocked reason, or None when launch is valid."""
    permitted_targets = profile["identity_match"].get("permitted_target_ids") or []
    if target_id not in permitted_targets:
        return f"target_id {target_id!r} is not permitted by the profile"
    snapshot_target = runtime_snapshot.get("target_id")
    if snapshot_target is not None and snapshot_target != target_id:
        return "runtime snapshot target_id does not match the requested target"
    device_arch = runtime_snapshot.get("device_arch")
    permitted_arches = profile["identity_match"].get("permitted_device_architectures") or []
    if device_arch is not None and permitted_arches and device_arch not in permitted_arches:
        return f"runtime device architecture {device_arch!r} is not permitted by the profile"
    if _resolve_interpreter(interpreter) is None:
        return f"interpreter {interpreter!r} is not available on this host"
    return None


def _write_blocked_run(
    *,
    run_dir: Path,
    profile: Mapping[str, Any],
    definition_file: Path,
    target_id: str,
    runtime_snapshot: Mapping[str, Any],
    definition: Mapping[str, Any],
    definition_sha256: str,
    run_id: str,
    reason: str,
) -> Path:
    started = datetime.now(timezone.utc)
    inputs: dict[str, dict[str, Any]] = {}
    inputs_root = run_dir / "inputs"
    profile_path = Path(profile["_profile_path"])
    _freeze_into(inputs, inputs_root, "profile.snapshot.yaml", profile_path)
    _freeze_into(inputs, inputs_root, "probe-definition.json", definition_file)
    result = {
        "schema_version": 1,
        "probe_id": definition["probe_id"],
        "implementation_profile_id": profile["implementation_profile_id"],
        "target_id": target_id,
        "profile_sha256": profile["_profile_sha256"],
        "definition_sha256": definition_sha256,
        "runtime_fingerprint": sha256_canonical_json(runtime_snapshot),
        "argv": [],
        "exit_code": None,
        "timed_out": False,
        "observed_scope": {},
        "observations": [],
        "evidence": [],
        "summary": "environment-blocked",
        "blocked_reason": reason,
    }
    write_json_atomic(run_dir / "results" / f"{definition['probe_id']}.json", result)
    ended = datetime.now(timezone.utc)
    run_document = {
        "schema_version": 1,
        "run_id": run_id,
        "target_id": target_id,
        "implementation_profile_id": profile["implementation_profile_id"],
        "implementation_profile_path": profile_path.as_posix(),
        "implementation_profile_version": profile["implementation_profile_version"],
        "implementation_profile_sha256": profile["_profile_sha256"],
        "runtime_fingerprint": sha256_canonical_json(runtime_snapshot),
        "requested_probe_ids": [definition["probe_id"]],
        "definition_sha256": definition_sha256,
        "start_timestamp": started.isoformat(),
        "end_timestamp": ended.isoformat(),
        "summary": "environment-blocked",
        "blocked_reason": reason,
        "inputs": inputs,
        "results": [_record(run_dir / "results" / f"{definition['probe_id']}.json", run_dir)],
        "evidence": [],
    }
    write_json_atomic(run_dir / "run.json", run_document)
    return run_dir


def _resolve_interpreter(interpreter: str) -> Path | None:
    if "/" in interpreter:
        candidate = Path(interpreter)
        return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None
    resolved = shutil.which(interpreter)
    return Path(resolved) if resolved else None


def _resolve_argv(
    argv: Sequence[str],
    *,
    definition: Mapping[str, Any],
    interpreter: str,
    probe_inputs_root: Path,
    probe_run_dir: Path,
    result_payload_path: Path,
    runtime_snapshot_path: Path,
    target_id: str,
    runtime_snapshot: Mapping[str, Any],
) -> list[str]:
    substitutions: dict[str, str] = {
        "interpreter": interpreter,
        "probe_inputs_root": str(probe_inputs_root),
        "probe_run_dir": str(probe_run_dir),
        "result_payload_path": str(result_payload_path),
        "runtime_snapshot_path": str(runtime_snapshot_path),
        "target_id": target_id,
    }
    for field in definition["required_runtime_fields"]:
        substitutions[field] = str(runtime_snapshot[field])
    resolved: list[str] = []
    for argument in argv:
        remaining = set(re.findall(r"\{([A-Za-z0-9_.-]+)\}", argument))
        unresolved = remaining - set(substitutions)
        if unresolved:
            raise _error("probe-placeholder-unresolved", f"unresolved argv placeholders {sorted(unresolved)!r}")
        resolved.append(argument.format_map(substitutions))
    return resolved


def _build_result(
    *,
    definition: Mapping[str, Any],
    definition_sha256: str,
    profile: Mapping[str, Any],
    target_id: str,
    runtime_snapshot: Mapping[str, Any],
    argv: Sequence[str],
    exit_code: int | None,
    timed_out: bool,
    payload: dict[str, Any] | None,
    evidence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if payload is not None:
        _validate_payload(payload, definition, profile, target_id)
        observations = payload.get("observations") or []
        observed_scope = payload.get("observed_scope") or {}
    else:
        observations = []
        observed_scope = {}
    return {
        "schema_version": 1,
        "probe_id": definition["probe_id"],
        "implementation_profile_id": profile["implementation_profile_id"],
        "target_id": target_id,
        "profile_sha256": profile["_profile_sha256"],
        "definition_sha256": definition_sha256,
        "runtime_fingerprint": sha256_canonical_json(runtime_snapshot),
        "argv": list(argv),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "observed_scope": observed_scope,
        "observations": observations,
        "evidence": list(evidence),
    }


def _validate_payload(payload: Mapping[str, Any], definition: Mapping[str, Any], profile: Mapping[str, Any], target_id: str) -> None:
    if payload.get("schema_version") != 1:
        raise _error("probe-payload-invalid", "probe payload schema_version must be 1")
    if payload.get("probe_id") != definition["probe_id"]:
        raise _error("probe-payload-invalid", "probe payload probe_id does not match the definition")
    if payload.get("implementation_profile_id") != profile["implementation_profile_id"]:
        raise _error("probe-payload-invalid", "probe payload implementation_profile_id does not match")
    if payload.get("target_id") != target_id:
        raise _error("probe-payload-invalid", "probe payload target_id does not match")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise _error("probe-payload-invalid", "probe payload requires at least one observation")
    observed_ids = set()
    for observation in observations:
        if not isinstance(observation, dict) or not isinstance(observation.get("capability_id"), str):
            raise _error("probe-payload-invalid", "each observation requires a capability_id")
        observed_ids.add(observation["capability_id"])
        if observation.get("level") not in {"observed", "inferred", "unknown"}:
            raise _error("probe-payload-invalid", "observation level must be observed|inferred|unknown")
        if not isinstance(observation.get("numerically_checked"), bool):
            raise _error("probe-payload-invalid", "observation requires numerically_checked boolean")
    declared = set(definition.get("capability_ids") or [])
    if not declared.issubset(observed_ids):
        raise _error("probe-payload-incomplete", "probe payload must observe every declared capability id")
    if not isinstance(payload.get("observed_scope"), dict):
        raise _error("probe-payload-invalid", "probe payload requires an observed_scope object")


def _classify(definition: Mapping[str, Any], exit_code: int | None, timed_out: bool, payload: dict[str, Any] | None) -> str:
    if timed_out:
        return "probe-failed"
    if exit_code is None:
        return "environment-blocked"
    if exit_code != 0 or payload is None:
        return "probe-failed"
    observations = payload.get("observations") or []
    if all(
        observation.get("level") == "observed" and observation.get("numerically_checked") is True
        for observation in observations
    ) and len(observations) == len(definition.get("capability_ids") or []):
        return "evidence-ready"
    return "partial"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="absolute canonical profile.yaml")
    parser.add_argument("--probe-id", required=True, help="catalog probe id")
    parser.add_argument("--target-id", required=True, help="concrete deployment target id")
    parser.add_argument("--runtime-snapshot", type=Path, required=True, help="absolute runtime-snapshot.json")
    parser.add_argument("--output-root", type=Path, required=True, help="absolute project-local output root")
    parser.add_argument("--qualification-requirement", type=Path, help="absolute normalized qualification requirement")
    parser.add_argument("--run-id", help="explicit safe run id; defaults to a UTC timestamp")
    args = parser.parse_args(argv)

    try:
        run_dir = run_profile_probe(
            profile_path=args.profile,
            probe_id=args.probe_id,
            target_id=args.target_id,
            runtime_snapshot_path=args.runtime_snapshot,
            output_root=args.output_root,
            qualification_requirement_path=args.qualification_requirement,
            run_id=args.run_id,
        )
    except (ContractValidationError, ProfileValidationError, ProbeValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"run_dir": str(run_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Deterministically derive a proposed profile-promotion candidate and note.

Never edits the canonical profile. The Markdown note is rendered from the
candidate JSON so the JSON remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from vnext_common import (
    ContractValidationError,
    load_json_document,
    load_json_yaml_document,
    sha256_file,
    write_json_atomic,
)
from validate_profile import ProfileValidationError, load_profile
from validate_probe import ProbeValidationError, validate_probe_run

V1_RECOMMENDABLE = frozenset({"constrained", "unknown", "unsupported", "prohibited"})


def _error(code: str, message: str, path: Path | None = None) -> ProbeValidationError:
    return ProbeValidationError(code, message, path)


def render_profile_promotion(run_dir: Path, *, profile_path: Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    profile_path = Path(profile_path)
    validation = validate_probe_run(run_dir)
    run = validation["run"]
    summary = validation["summary"]
    if summary in {"environment-blocked", "probe-failed"}:
        raise _error("promotion-unavailable", f"cannot promote a {summary} probe run", run_dir)
    result_path = run_dir / "results" / f"{run['requested_probe_ids'][0]}.json"
    result = load_json_document(result_path, artifact="probe result")
    observed_facts = [
        observation
        for observation in result.get("observations") or []
        if observation.get("level") == "observed" and observation.get("numerically_checked") is True
    ]
    if summary == "partial" and not observed_facts:
        raise _error("promotion-no-observed-fact", "a partial run with no observed fact cannot be promoted", run_dir)

    current_profile = load_profile(profile_path)
    frozen_profile = load_json_yaml_document(run_dir / "inputs" / "profile.snapshot.yaml", artifact="frozen profile")
    runtime_snapshot = load_json_document(run_dir / "inputs" / "runtime-snapshot.json", artifact="runtime snapshot")
    definition = load_json_document(run_dir / "inputs" / "probe-definition.json", artifact="probe definition")
    if frozen_profile.get("implementation_profile_id") != current_profile["implementation_profile_id"]:
        raise _error("promotion-profile-mismatch", "frozen run profile does not match the current implementation profile", run_dir)

    recommendations: list[dict[str, Any]] = []
    unresolved_gaps: list[str] = []
    for observation in result.get("observations") or []:
        capability_id = observation["capability_id"]
        capability = next(
            (entry for entry in current_profile["capability_matrix"] if entry["id"] == capability_id),
            None,
        )
        if capability is None:
            unresolved_gaps.append(f"capability {capability_id!r} is not present in the current profile")
            continue
        current_status = capability["status"]
        if observation.get("level") == "observed" and observation.get("numerically_checked") is True:
            recommended_status = _recommended_status(current_status)
            evidence_refs = [
                result_path.relative_to(run_dir).as_posix(),
                (run_dir / "run.json").relative_to(run_dir).as_posix(),
            ]
            source_scope = _source_scope(result, run, runtime_snapshot)
            _validate_recommendation_scope(capability_id, source_scope, result, run, runtime_snapshot)
            recommendations.append(
                {
                    "capability_id": capability_id,
                    "current_status": current_status,
                    "recommended_status": recommended_status,
                    "source_scope": source_scope,
                    "evidence_refs": evidence_refs,
                    "rationale": (
                        f"Numerically checked observed success within the probe's exact scope; "
                        f"v1 renderer never recommends supported."
                    ),
                }
            )
        else:
            recommendations.append(
                {
                    "capability_id": capability_id,
                    "current_status": current_status,
                    "recommended_status": current_status,
                    "source_scope": _source_scope(result, run, runtime_snapshot),
                    "evidence_refs": [],
                    "rationale": "No numerically checked observed success; status unchanged and evidence gap recorded.",
                }
            )
            unresolved_gaps.append(f"capability {capability_id!r} has no numerically checked observed success")

    candidate = {
        "schema_version": 1,
        "review_status": "proposed",
        "implementation_profile_id": run["implementation_profile_id"],
        "probe_id": definition["probe_id"],
        "probe_definition_sha256": result["definition_sha256"],
        "result_sha256": sha256_file(result_path),
        "run_id": run["run_id"],
        "recommendations": recommendations,
        "unresolved_gaps": unresolved_gaps,
    }
    if (run_dir / "inputs" / "qualification-requirement.json").is_file() and observed_facts:
        candidate["onboarding_disposition"] = "promotion-pending"

    validate_promotion_candidate_path = _write_candidate(run_dir, candidate, current_profile, run, result, runtime_snapshot)
    note_path = run_dir / "promotion-note.md"
    note_path.write_text(_render_note(candidate), encoding="utf-8")
    return validate_promotion_candidate_path, note_path


def _recommended_status(current_status: str) -> str:
    if current_status == "unknown":
        return "constrained"
    return current_status


def _source_scope(result: Mapping[str, Any], run: Mapping[str, Any], runtime_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    scope: dict[str, Any] = {"target_id": run["target_id"]}
    if isinstance(runtime_snapshot.get("device_arch"), str):
        scope["device_arch"] = runtime_snapshot["device_arch"]
    if isinstance(runtime_snapshot.get("toolchain"), str):
        scope["toolchain"] = runtime_snapshot["toolchain"]
    observed_scope = result.get("observed_scope")
    if isinstance(observed_scope, dict):
        scope.update(observed_scope)
    return scope


def _validate_recommendation_scope(
    capability_id: str,
    source_scope: Mapping[str, Any],
    result: Mapping[str, Any],
    run: Mapping[str, Any],
    runtime_snapshot: Mapping[str, Any],
) -> None:
    observed_scope = result.get("observed_scope") if isinstance(result.get("observed_scope"), dict) else {}
    for key, value in source_scope.items():
        if key in observed_scope:
            if observed_scope[key] != value:
                raise _error("promotion-scope-widened", f"recommendation scope for {capability_id!r} is broader than the observed scope")
        elif key == "target_id":
            if run.get("target_id") != value:
                raise _error("promotion-scope-widened", f"recommendation target scope for {capability_id!r} exceeds the run target")
        elif key in {"device_arch", "toolchain"}:
            if runtime_snapshot.get(key) != value:
                raise _error("promotion-scope-widened", f"recommendation {key} scope for {capability_id!r} exceeds the runtime identity")


def _write_candidate(
    run_dir: Path,
    candidate: Mapping[str, Any],
    profile: Mapping[str, Any],
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    runtime_snapshot: Mapping[str, Any],
) -> Path:
    candidate_path = run_dir / "promotion-candidate.json"
    validate_promotion_candidate(candidate, run_dir=run_dir, profile=profile, run=run, result=result, runtime_snapshot=runtime_snapshot)
    write_json_atomic(candidate_path, candidate)
    return candidate_path


def validate_promotion_candidate(
    candidate: Mapping[str, Any] | Path,
    *,
    run_dir: Path,
    profile: Mapping[str, Any],
    run: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
    runtime_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a proposed promotion candidate against its run and profile."""
    if isinstance(candidate, (str, Path)):
        candidate = load_json_document(Path(candidate), artifact="promotion candidate")
    run_dir = Path(run_dir)
    if run is None:
        run = load_json_document(run_dir / "run.json", artifact="probe run")
    if result is None:
        result = load_json_document(run_dir / "results" / f"{run['requested_probe_ids'][0]}.json", artifact="probe result")
    if runtime_snapshot is None:
        runtime_snapshot = load_json_document(run_dir / "inputs" / "runtime-snapshot.json", artifact="runtime snapshot")

    if candidate.get("schema_version") != 1:
        raise _error("promotion-schema-version", "promotion candidate schema_version must be 1")
    if candidate.get("review_status") != "proposed":
        raise _error("promotion-review-status", "promotion candidate review_status must be proposed")
    if candidate.get("implementation_profile_id") != profile["implementation_profile_id"]:
        raise _error("promotion-profile-mismatch", "promotion candidate implementation_profile_id must match the profile")
    if candidate.get("probe_id") != run["requested_probe_ids"][0]:
        raise _error("promotion-probe-mismatch", "promotion candidate probe_id must match the run")
    if candidate.get("probe_definition_sha256") != result.get("definition_sha256"):
        raise _error("promotion-definition-hash", "promotion candidate definition hash must match the result")
    if candidate.get("run_id") != run.get("run_id"):
        raise _error("promotion-run-mismatch", "promotion candidate run_id must match the run")

    result_hash = sha256_file(run_dir / "results" / f"{run['requested_probe_ids'][0]}.json")
    if candidate.get("result_sha256") != result_hash:
        raise _error("promotion-result-hash", "promotion candidate result hash must match the run result")

    observed_scope = result.get("observed_scope") if isinstance(result.get("observed_scope"), dict) else {}
    profile_ids = {entry["id"] for entry in profile["capability_matrix"]}
    recommendations = candidate.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        raise _error("promotion-recommendations", "promotion candidate requires recommendations")
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise _error("promotion-recommendations", "each recommendation must be an object")
        capability_id = recommendation.get("capability_id")
        if capability_id not in profile_ids:
            raise _error("promotion-capability-unknown", f"recommendation names unknown capability {capability_id!r}")
        recommended = recommendation.get("recommended_status")
        if recommended == "supported":
            raise _error("promotion-supported-forbidden", "the v1 renderer never recommends supported")
        if recommended not in V1_RECOMMENDABLE:
            raise _error("promotion-status-invalid", f"recommended status {recommended!r} is invalid")
        current = recommendation.get("current_status")
        if current not in V1_RECOMMENDABLE:
            raise _error("promotion-status-invalid", f"current status {current!r} is invalid")
        if recommended != current and recommended != "constrained":
            raise _error("promotion-status-invalid", "v1 recommendations may only constrain or keep the status")
        source_scope = recommendation.get("source_scope")
        if not isinstance(source_scope, dict):
            raise _error("promotion-scope-invalid", "each recommendation requires a source_scope object")
        for key, value in source_scope.items():
            if key in observed_scope:
                if observed_scope[key] != value:
                    raise _error("promotion-scope-widened", f"recommendation scope for {capability_id!r} is broader than the observed scope")
            elif key == "target_id" and run.get("target_id") != value:
                raise _error("promotion-scope-widened", f"recommendation target scope for {capability_id!r} exceeds the run target")
            elif key in {"device_arch", "toolchain"} and runtime_snapshot.get(key) != value:
                raise _error("promotion-scope-widened", f"recommendation {key} scope for {capability_id!r} exceeds the runtime identity")

    disposition = candidate.get("onboarding_disposition")
    if disposition is not None and disposition != "promotion-pending":
        raise _error("promotion-disposition-invalid", "onboarding_disposition may only be promotion-pending")
    if disposition == "promotion-pending" and not (run_dir / "inputs" / "qualification-requirement.json").is_file():
        raise _error("promotion-disposition-invalid", "promotion-pending requires a before-fallback qualification requirement in the run")
    return {"valid": True, "candidate": candidate}


def _render_note(candidate: Mapping[str, Any]) -> str:
    lines = [
        "# Proposed profile promotion",
        "",
        f"- Review status: `{candidate['review_status']}`",
        f"- Implementation profile: `{candidate['implementation_profile_id']}`",
        f"- Probe: `{candidate['probe_id']}` (definition `{candidate['probe_definition_sha256'][:12]}…`, result `{candidate['result_sha256'][:12]}…`)",
        f"- Run: `{candidate['run_id']}`",
    ]
    if candidate.get("onboarding_disposition"):
        lines.append(f"- Onboarding disposition: `{candidate['onboarding_disposition']}`")
    lines.extend(["", "## Recommendations", ""])
    for recommendation in candidate["recommendations"]:
        lines.append(
            f"- `{recommendation['capability_id']}`: "
            f"`{recommendation['current_status']}` -> `{recommendation['recommended_status']}`"
        )
        lines.append(f"  - Scope: `{json.dumps(recommendation['source_scope'], sort_keys=True)}`")
        lines.append(f"  - Rationale: {recommendation['rationale']}")
    gaps = candidate.get("unresolved_gaps") or []
    if gaps:
        lines.extend(["", "## Unresolved gaps", ""])
        lines.extend(f"- {gap}" for gap in gaps)
    lines.extend(
        [
            "",
            "This note is a rendering of `promotion-candidate.json`, which remains authoritative. "
            "It never edits the canonical profile; promotion requires an explicit maintainer review commit.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="absolute completed probe run directory")
    parser.add_argument("--profile", type=Path, required=True, help="absolute current canonical profile.yaml")
    args = parser.parse_args(argv)

    try:
        candidate_path, note_path = render_profile_promotion(args.run_dir, profile_path=args.profile)
    except (ContractValidationError, ProfileValidationError, ProbeValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps([str(candidate_path), str(note_path)], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

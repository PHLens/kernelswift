from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from campaign_import import ValidatedCampaign
from kernelwiki_common import KernelWikiError, canonical_json_bytes, sha256_bytes
from lift_schema import validate_lift_document


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "schemas.yaml"
_FORBIDDEN_KEYS = {"next_candidate", "recommended_next_change", "implementation_instruction"}


@dataclass(frozen=True)
class ExperienceProposal:
    schema_version: int
    proposal_id: str
    source_lane: str
    contract_version: int
    loop_contract_identity: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]
    terminal: Mapping[str, Any]
    scope: Mapping[str, Any]
    expected: Mapping[str, Any]
    observed: tuple[Mapping[str, Any], ...]
    suggested_publication: Mapping[str, Any]
    transfer_boundaries: tuple[str, ...]
    reconsider_when: tuple[str, ...]
    missing_evidence: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        value = asdict(self)
        value["observed"] = list(self.observed)
        value["transfer_boundaries"] = list(self.transfer_boundaries)
        value["reconsider_when"] = list(self.reconsider_when)
        value["missing_evidence"] = list(self.missing_evidence)
        return value


def _identity(validated: ValidatedCampaign) -> dict[str, Any]:
    identity = validated.loop_contract_identity
    return {
        "repository_commit": identity.repository_commit,
        "skill_tree_sha": identity.skill_tree_sha,
        "validator_sha256": dict(sorted(identity.validator_sha256.items())),
        "schema_sha256": dict(sorted(identity.schema_sha256.items())),
    }


def _proposal_id(validated: ValidatedCampaign) -> str:
    identity = {
        "contract_version": validated.bundle.contract_version,
        "terminal_commit": validated.bundle.terminal_commit,
        "round_id": validated.bundle.round_id,
        "terminal_result": validated.bundle.terminal_result,
    }
    return "experience-" + sha256_bytes(canonical_json_bytes(identity))[:20]


def _claim(validated: ValidatedCampaign) -> Mapping[str, Any]:
    claim = validated.normalized_claim.get("claim", validated.normalized_claim)
    return claim if isinstance(claim, Mapping) else {}


def _sketch(validated: ValidatedCampaign) -> Mapping[str, Any]:
    sketch = validated.normalized_sketch.get("sketch", validated.normalized_sketch)
    return sketch if isinstance(sketch, Mapping) else {}


def _scope(validated: ValidatedCampaign, measurement_fingerprint: str | None) -> dict[str, Any]:
    claim = _claim(validated)
    profile = validated.normalized_profile
    sketch = _sketch(validated)
    implementation = profile.get("implementation", {})
    declarations = sketch.get("declarations", [])
    dtypes = sorted({item["dtype"] for item in declarations if isinstance(item, Mapping) and isinstance(item.get("dtype"), str)})
    shapes = sorted({"x".join(map(str, item["shape"])) for item in declarations if isinstance(item, Mapping) and isinstance(item.get("shape"), list)})
    device_arches = profile.get("identity_match", {}).get("permitted_device_architectures", [])
    return {
        "target_id": claim.get("target_id"),
        "implementation_profile_id": profile.get("implementation_profile_id"),
        "implementation_profile_version": profile.get("implementation_profile_version"),
        "profile_status": profile.get("profile_status"),
        "runtime_fingerprint": claim.get("runtime_fingerprint"),
        "device_architectures": sorted(device_arches) if isinstance(device_arches, list) else [],
        "language": implementation.get("language") if isinstance(implementation, Mapping) else None,
        "backend": implementation.get("backend") if isinstance(implementation, Mapping) else None,
        "shape_signatures": shapes,
        "dtypes": dtypes,
        "measurement_fingerprint": measurement_fingerprint,
        "comparability": validated.fact_pack.get("comparability"),
    }


def _expected(validated: ValidatedCampaign) -> dict[str, Any]:
    decision = validated.normalized_decision
    intent = decision.get("optimization_intent", {})
    evaluation = decision.get("evaluation_contract", {})
    graph = evaluation.get("causal_graph", {}) if isinstance(evaluation, Mapping) else {}
    nodes = graph.get("nodes", []) if isinstance(graph, Mapping) else []
    sketch = _sketch(validated)
    operations = sketch.get("operations", [])
    return {
        "intervention": intent.get("intervention") if isinstance(intent, Mapping) else None,
        "bottleneck_class": intent.get("bottleneck_class") if isinstance(intent, Mapping) else None,
        "expected_wall_improvement_pct": intent.get("expected_wall_improvement_pct") if isinstance(intent, Mapping) else None,
        "causal_observables": sorted(item for item in nodes if isinstance(item, str) and item.startswith("o.")),
        "sketch_statement_ids": sorted(item["id"] for item in operations if isinstance(item, Mapping) and isinstance(item.get("id"), str)),
    }


def _record(metric: str, value: Any, statistic: str, unit: str) -> dict[str, Any]:
    return {"metric": metric, "value": value, "statistic": statistic, "unit": unit}


def _observed(validated: ValidatedCampaign) -> tuple[Mapping[str, Any], ...]:
    facts = validated.fact_pack
    records: list[dict[str, Any]] = []
    raw_measurements = facts.get("measurements", [])
    if isinstance(raw_measurements, list):
        for item in raw_measurements:
            if not isinstance(item, Mapping):
                continue
            if all(key in item for key in ("metric", "value", "statistic", "unit")):
                records.append(_record(str(item["metric"]), item["value"], str(item["statistic"]), str(item["unit"])))
    for label in ("correctness", "lowering"):
        value = facts.get(label)
        if isinstance(value, Mapping) and value.get("status") is not None:
            records.append(_record(label, value["status"], "status", "state"))
    for label in ("kernel_count", "device_time", "wall_time"):
        value = facts.get(label)
        if isinstance(value, Mapping) and value.get("value") is not None:
            records.append(_record(label, value["value"], str(value.get("statistic", "value")), str(value.get("unit", "unknown"))))
    profiler = facts.get("profiler")
    if isinstance(profiler, Mapping):
        for name, value in sorted(profiler.items()):
            if isinstance(value, (str, int, float, bool)) or value is None:
                records.append(_record(f"profiler.{name}", value, "observed", "state"))
    capability_status = facts.get("required_capability_status")
    if capability_status is not None:
        records.append(_record("required_capability_status", capability_status, "status", "state"))
    observables = facts.get("observables", [])
    if isinstance(observables, list):
        for item in observables:
            if isinstance(item, Mapping) and isinstance(item.get("name"), str):
                records.append(_record(item["name"], item.get("value"), str(item.get("status", "observed")), str(item.get("unit", "state"))))
    unique = {canonical_json_bytes(record): record for record in records}
    return tuple(unique[key] for key in sorted(unique))


def _measurement_value(observed: tuple[Mapping[str, Any], ...], metric: str) -> float | None:
    for record in observed:
        if record.get("metric") != metric:
            continue
        value = record.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _publication_mapping(validated: ValidatedCampaign, observed: tuple[Mapping[str, Any], ...], fingerprint: str | None) -> dict[str, Any]:
    facts = validated.fact_pack
    verdict = validated.normalized_verdict
    terminal = str(verdict.get("terminal_result") or validated.bundle.terminal_result).lower()
    classification = str(verdict.get("classification") or "none").lower()
    route = str(verdict.get("route") or validated.normalized_decision.get("metadata", {}).get("decision") or "").lower()
    comparability = facts.get("comparability")
    comparable = comparability is True or str(comparability).lower() == "comparable"
    performance_status = str(facts.get("performance_status") or "").lower()
    missing_text = " ".join(validated.missing_evidence).lower()
    qualification = " ".join(map(str, _claim(validated).get("qualification_dispositions", []))).lower()
    capability_status = str(facts.get("required_capability_status") or "").lower()
    device_delta = _measurement_value(observed, "device_time")
    wall_delta = _measurement_value(observed, "wall_time")

    def result(decision: str, role: str | None = None, subtype: str | None = None, *card_ids: str) -> dict[str, Any]:
        return {
            "decision": decision,
            "mode": "existing-card-example" if decision == "include" else None,
            "role": role,
            "subtype": subtype,
            "card_ids": sorted(card_ids),
            "tags": sorted(item for item in {role, subtype} if item),
        }

    if fingerprint is None:
        return result("defer")
    if capability_status in {"unknown", "unsupported"} or any(word in qualification for word in ("unknown", "unsupported", "capability")):
        return result("include", "capability-gap", "profile", "pattern-ascend-capability-gap")
    if terminal in {"probe-only", "environment-blocked", "incomplete"} or any(word in missing_text for word in ("incomplete", "probe-required")):
        return result("defer")
    if "semantic" in classification or (route == "abort" and "design" in classification):
        return result("include", "counterexample", "design-pitfall")
    if "implementation" in classification or "coder" in classification or terminal in {"coder-failed", "implementation-failed"}:
        return result("include", "counterexample", "implementation-pitfall")
    if terminal == "screened-out":
        return result("include", "counterexample", "screening")
    if device_delta is not None and wall_delta is not None and device_delta < 0 < wall_delta:
        return result("include", "counterexample", "device-wall-mismatch", "pattern-device-win-wall-loss")
    if terminal == "no-improvement" or performance_status in {"slower", "no-improvement", "regressed"} or (wall_delta is not None and wall_delta >= 0):
        return result("include", "counterexample", "performance")
    if terminal == "accepted" and comparable and (performance_status in {"improved", "faster"} or (wall_delta is not None and wall_delta < 0)):
        change_family = validated.normalized_decision.get("metadata", {}).get("change_family")
        cards = ("technique-kernel-fusion",) if change_family == "kernel-fusion" else ()
        return result("include", "positive", "performance", *cards)
    return result("defer")


def _walk_forbidden(value: Any) -> None:
    if isinstance(value, Mapping):
        found = _FORBIDDEN_KEYS & set(value)
        if found:
            raise KernelWikiError("proposal-forbidden", f"proposal contains forbidden fields: {', '.join(sorted(found))}")
        for nested in value.values():
            _walk_forbidden(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _walk_forbidden(nested)


def build_experience_proposal(validated: ValidatedCampaign) -> ExperienceProposal:
    if validated.bundle.contract_version != 3:
        raise KernelWikiError("contract-unsupported", f"unsupported experience mapping contract: {validated.bundle.contract_version}")
    fingerprint_value = validated.fact_pack.get("measurement_fingerprint")
    fingerprint = fingerprint_value if isinstance(fingerprint_value, str) and fingerprint_value else None
    missing = set(validated.missing_evidence)
    if fingerprint is None:
        missing.add("measurement-fingerprint-missing")
    observed = _observed(validated)
    publication = _publication_mapping(validated, observed, fingerprint)
    scope = _scope(validated, fingerprint)
    claim = _claim(validated)
    transfer = (
        f"target={claim.get('target_id') or 'unknown'}",
        f"profile={validated.normalized_profile.get('implementation_profile_id') or 'unknown'}",
        f"runtime={claim.get('runtime_fingerprint') or 'unknown'}",
        "measurements apply only to the recorded fingerprint and comparability scope",
    )
    reconsider = tuple(sorted({"target, profile, runtime, shape, or dtype scope changes", *(f"evidence becomes available: {item}" for item in missing)}))
    proposal = ExperienceProposal(
        schema_version=1,
        proposal_id=_proposal_id(validated),
        source_lane="strict-current-vnext",
        contract_version=validated.bundle.contract_version,
        loop_contract_identity=_identity(validated),
        artifact_hashes=dict(sorted(validated.artifact_hashes.items())),
        terminal={
            "round_id": validated.bundle.round_id,
            "result": validated.bundle.terminal_result,
            "commit": validated.bundle.terminal_commit,
            "classification": validated.normalized_verdict.get("classification"),
            "route": validated.normalized_verdict.get("route") or validated.normalized_decision.get("metadata", {}).get("decision"),
            "attribution": {
                "failed_attempt_effect": validated.normalized_verdict.get("failed_attempt_effect"),
                "evidence_gap_cause": validated.fact_pack.get("evidence_gap_cause"),
            },
        },
        scope=scope,
        expected=_expected(validated),
        observed=observed,
        suggested_publication=publication,
        transfer_boundaries=transfer,
        reconsider_when=reconsider,
        missing_evidence=tuple(sorted(missing)),
    )
    document = proposal.to_document()
    _walk_forbidden(document)
    validate_lift_document("experience_proposal", document, SCHEMA_PATH)
    return proposal


def write_proposal(proposal: ExperienceProposal, output_path: Path) -> str:
    path = Path(output_path)
    document = proposal.to_document()
    _walk_forbidden(document)
    validate_lift_document("experience_proposal", document, SCHEMA_PATH, path=path)
    data = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as error:
        raise KernelWikiError("proposal-exists", "proposal output already exists", path) from error
    return sha256_bytes(data)

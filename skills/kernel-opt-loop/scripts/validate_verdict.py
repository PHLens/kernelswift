"""Causal graph, Verifier fact pack, deterministic final-tuning selection, and
attribution verdict validation.

The verdict validator must never let free-form explanation select a class; only
satisfied rule preconditions may produce a classification. Finalization verdicts
use a separate counter-free branch.
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
    load_json_document,
    sha256_canonical_json,
)
from validate_decision import extract_sections, parse_single_json_block


class VerdictValidationError(ContractValidationError):
    pass


def _error(code: str, message: str, path: Path | None = None) -> VerdictValidationError:
    return VerdictValidationError(code, message, path)


RULES = {
    "DESIGN.SKETCH.INVALID": {"classification": "design-error", "terminal_result": "design-rejected", "failed_attempt_effect": "increment"},
    "DESIGN.CAUSAL.INVALID": {"classification": "design-error", "terminal_result": "design-rejected", "failed_attempt_effect": "increment"},
    "CODE.BINDING.MISSING": {"classification": "code-error", "terminal_result": "candidate-failed", "failed_attempt_effect": "increment", "repairable": True},
    "CODE.BINDING.VIOLATION": {"classification": "code-error", "terminal_result": "candidate-failed", "failed_attempt_effect": "increment", "repairable": True},
    "CODE.CORRECTNESS.FAIL": {"classification": "code-error", "terminal_result": "candidate-failed", "failed_attempt_effect": "increment"},
    "LOWERING.EXPECTED.ABSENT": {"classification": "lowering-unknown", "terminal_result": "design-rejected", "failed_attempt_effect": "unchanged"},
    "EVIDENCE.OBSERVABLE.MISSING": {
        "classification": "evidence-gap",
        "terminal_result": "blocked",
        "failed_attempt_effect": "unchanged",
        "cause_overrides": {
            "decision": {"classification": "design-error", "terminal_result": "design-rejected", "failed_attempt_effect": "increment"}
        },
    },
}

RULE_PRECONDITIONS = {
    "DESIGN.SKETCH.INVALID": ("sketch_valid",),
    "DESIGN.CAUSAL.INVALID": ("causal_graph_valid",),
    "CODE.BINDING.MISSING": ("binding_valid",),
    "CODE.BINDING.VIOLATION": ("binding_valid",),
    "CODE.CORRECTNESS.FAIL": ("correctness_pass",),
    "LOWERING.EXPECTED.ABSENT": ("binding_valid", "correctness_pass", "lowering_observed", "expected_mechanism_absent"),
    "EVIDENCE.OBSERVABLE.MISSING": ("bounded_probe_attempted", "observable_observed"),
}

VALID_OBSERVABLE_STATUSES = frozenset({"observed", "missing", "unavailable", "inconclusive"})


def validate_causal_graph(graph: Mapping[str, Any], *, intervention: str, observable_names: Sequence[str]) -> None:
    """Validate structural connectivity of a causal graph around one intervention."""
    if not isinstance(graph, dict):
        raise _error("causal-graph-invalid", "causal graph must be an object")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes or any(not isinstance(node, str) or not node for node in nodes):
        raise _error("causal-graph-invalid", "causal graph nodes must be a nonempty string list")
    if len(set(nodes)) != len(nodes):
        raise _error("causal-graph-invalid", "causal graph nodes must be unique")
    if not isinstance(edges, list):
        raise _error("causal-graph-invalid", "causal graph edges must be a list")
    node_set = set(nodes)
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or edge[0] not in node_set or edge[1] not in node_set:
            raise _error("causal-graph-invalid", "each causal edge must name two existing nodes")
    if not intervention:
        raise _error("causal-graph-invalid", "causal graph requires a nonempty intervention")
    if not observable_names or not all(_observable_in_graph(name, node_set) for name in observable_names):
        raise _error("causal-graph-disconnected", "every declared observable must be a causal graph node")

    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for edge in edges:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    visited: set[str] = set()
    stack = [nodes[0]]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency[node] - visited)
    if len(visited) != len(nodes):
        raise _error("causal-graph-disconnected", "causal graph is disconnected")


def extract_verifier_fact_pack(report_path: Path) -> dict[str, Any]:
    """Extract and normalize the single vNext fact-pack JSON fence from a report."""
    report_path = Path(report_path)
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError as error:
        raise _error("report-read-error", f"cannot read report {report_path}", report_path) from error
    sections = extract_sections(text)
    section = sections.get("vNext Fact Pack")
    if section is None:
        raise _error("fact-pack-section-missing", "report requires a ## vNext Fact Pack section", report_path)
    match = re.fullmatch(
        r"[ \t\r\n]*```json[ \t]*\r?\n(.*?)\r?\n```[ \t]*[\r\n]*",
        section.body,
        flags=re.DOTALL,
    )
    if not match:
        raise _error("fact-pack-json-required", "vNext Fact Pack must contain exactly one fenced json object", report_path)
    try:
        facts = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise _error("fact-pack-json-invalid", f"vNext Fact Pack JSON is invalid: {error.msg}", report_path) from error
    if not isinstance(facts, dict):
        raise _error("fact-pack-object-required", "vNext Fact Pack JSON must be an object", report_path)
    _validate_fact_pack(facts)
    return facts


def _observable_in_graph(name: str, node_set: set[str]) -> bool:
    return name in node_set or any(node.endswith(f".{name}") for node in node_set)


def _validate_fact_pack(facts: Mapping[str, Any]) -> None:
    if facts.get("schema_version") != 1:
        raise _error("fact-pack-schema-version", "fact pack schema_version must be 1")
    if not isinstance(facts.get("candidate_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", facts["candidate_sha256"]):
        raise _error("fact-pack-candidate-hash", "fact pack requires candidate_sha256")
    correctness = facts.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("status") not in {"pass", "fail"}:
        raise _error("fact-pack-correctness", "fact pack correctness status must be pass or fail")
    observables = facts.get("observables")
    if not isinstance(observables, list) or not observables:
        raise _error("fact-pack-observables", "fact pack requires observables")
    for observable in observables:
        if not isinstance(observable, dict) or not isinstance(observable.get("name"), str) or not observable["name"]:
            raise _error("fact-pack-observables", "each observable requires a name")
        if observable.get("status") not in VALID_OBSERVABLE_STATUSES:
            raise _error("fact-pack-observables", "observable status must be observed|missing|unavailable|inconclusive")
    lowering = facts.get("lowering")
    if not isinstance(lowering, dict) or lowering.get("status") not in {"observed", "unavailable"}:
        raise _error("fact-pack-lowering", "fact pack lowering status must be observed or unavailable")
    if facts.get("evidence_gap_cause") not in {"none", "environment", "decision"}:
        raise _error("fact-pack-gap-cause", "fact pack evidence_gap_cause must be none|environment|decision")


def select_final_tuning_configuration(
    contract: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pure deterministic selector over normalized search trials."""
    configurations = contract.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise _error("final-tuning-contract", "final tuning contract requires configurations")
    declared = [json.dumps(configuration, sort_keys=True) for configuration in configurations]
    declared_set = set(declared)
    fallback = contract.get("fallback_configuration")
    if not isinstance(fallback, dict):
        raise _error("final-tuning-contract", "final tuning contract requires fallback_configuration")
    fallback_key = json.dumps(fallback, sort_keys=True)
    if fallback_key not in declared_set:
        raise _error("final-tuning-contract", "fallback/control configuration must be declared")
    if not isinstance(trials, list):
        raise _error("final-tuning-trials", "search trials must be a list")
    if len(trials) > contract.get("max_trials", len(trials)):
        raise _error("final-tuning-budget", "search trials exceed the declared trial budget")

    by_config: dict[str, dict[str, Any]] = {}
    seen_order: set[int] = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise _error("final-tuning-trials", "each trial must be an object")
        order = trial.get("order_index")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise _error("final-tuning-trials", "each trial requires a non-negative order_index")
        if order in seen_order:
            raise _error("final-tuning-trials", "trial order indexes must be unique")
        seen_order.add(order)
        configuration = trial.get("configuration")
        if not isinstance(configuration, dict):
            raise _error("final-tuning-trials", "each trial requires a configuration")
        key = json.dumps(configuration, sort_keys=True)
        if key not in declared_set:
            raise _error("final-tuning-trials", "a trial configuration is not declared in the contract")
        if key in by_config:
            raise _error("final-tuning-trials", "duplicate trial configuration")
        by_config[key] = trial

    tie_rule = contract.get("tie_rule")
    if tie_rule != "first-in-declared-order":
        raise _error("final-tuning-tie", "only the first-in-declared-order tie rule is supported")
    metric = contract.get("comparison_metric")
    if not isinstance(metric, str) or not metric:
        raise _error("final-tuning-contract", "final tuning contract requires comparison_metric")

    eligible = [
        trial
        for trial in trials
        if trial.get("eligibility") is True
        and trial.get("compile_status") == "ok"
        and trial.get("correctness_status") == "pass"
        and trial.get("reset_status") == "ok"
        and isinstance(trial.get("statistic"), (int, float))
    ]
    if not eligible:
        return dict(fallback)

    def metric_value(trial: Mapping[str, Any]) -> float:
        return float(trial["statistic"])

    order_rank = {order: index for index, order in enumerate(sorted(seen_order))}
    fallback_trial = by_config.get(fallback_key)
    if fallback_trial is not None and fallback_trial.get("eligibility") is True and isinstance(fallback_trial.get("statistic"), (int, float)):
        fallback_value = metric_value(fallback_trial)
    else:
        fallback_value = None
    best = min(eligible, key=lambda trial: (metric_value(trial), order_rank[trial["order_index"]]))
    if fallback_value is not None and metric_value(best) >= fallback_value:
        return dict(fallback)
    return dict(best["configuration"])


@dataclass(frozen=True)
class FinalizationSlot:
    artifact_index: str
    resume_decision: Path | None = None
    resume_report: Path | None = None


def resolve_finalization_slot(project_root: Path, submission_snapshot_id: str) -> FinalizationSlot:
    """Allocate or resume the deterministic artifact index for one submission snapshot."""
    project_root = Path(project_root)
    rounds = project_root / "rounds"
    if not rounds.is_dir():
        raise _error("finalization-rounds-missing", "project rounds directory does not exist")

    indexed_files = []
    for pattern in ("decision_*.md", "binding_*.json", "report_*.md", "verdict_*.json"):
        indexed_files.extend(rounds.glob(pattern))
    occupied = sorted(
        {int(match.group(1)) for path in indexed_files if (match := re.fullmatch(r"(?:decision|binding|report|verdict)_([0-9]{3})\.[a-z]+", path.name))}
    )

    for verdict_path in rounds.glob("verdict_*.json"):
        verdict = load_json_document(verdict_path, artifact="verdict")
        if verdict.get("submission_snapshot_id") == submission_snapshot_id:
            raise _error("finalization-already-completed", f"submission snapshot {submission_snapshot_id[:12]}… is already finalized", verdict_path)

    resume_decision: Path | None = None
    resume_report: Path | None = None
    for decision_path in rounds.glob("decision_*.md"):
        try:
            sections = extract_sections(decision_path.read_text(encoding="utf-8"))
            metadata = parse_single_json_block(sections["Metadata"])
        except (ContractValidationError, KeyError, OSError):
            continue
        if metadata.get("decision_kind") != "final-autotune":
            continue
        index = metadata.get("artifact_index")
        if not isinstance(index, str) or not re.fullmatch(r"[0-9]{3}", index):
            continue
        tuning = _read_tuning_section(decision_path)
        if tuning is None or tuning.get("submission_snapshot_id") != submission_snapshot_id:
            if int(index) in occupied:
                raise _error("finalization-conflicting-decision", f"artifact index {index} is reserved by a conflicting decision", decision_path)
            continue
        candidate_report = rounds / f"report_{index}.md"
        if candidate_report.is_file():
            try:
                facts = extract_verifier_fact_pack(candidate_report)
                if facts.get("final_configuration_tuning", {}).get("submission_snapshot_id") == submission_snapshot_id:
                    resume_report = candidate_report
                    resume_decision = decision_path
                    continue
            except VerdictValidationError:
                pass
            raise _error("finalization-report-invalid", f"report {candidate_report.name} is incomplete or invalid", candidate_report)
        resume_decision = decision_path

    if resume_decision is not None:
        index = _decision_index(resume_decision)
        return FinalizationSlot(artifact_index=index, resume_decision=resume_decision, resume_report=resume_report)
    next_index = (max(occupied) + 1) if occupied else 1
    return FinalizationSlot(artifact_index=f"{next_index:03d}")


def _read_tuning_section(decision_path: Path) -> dict[str, Any] | None:
    sections = extract_sections(decision_path.read_text(encoding="utf-8"))
    tuning = sections.get("Final Configuration Tuning")
    if tuning is None:
        return None
    try:
        return parse_single_json_block(tuning)
    except ContractValidationError:
        return None


def _decision_index(decision_path: Path) -> str:
    match = re.fullmatch(r"decision_([0-9]{3})\.md", decision_path.name)
    return match.group(1) if match else "001"


def apply_submission_promotion(state: Mapping[str, Any], verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Pure atomic update of the accepted kernel/report submission pair."""
    if not isinstance(state, Mapping):
        raise _error("submission-state-invalid", "submission state must be a mapping")
    if verdict.get("route") not in {"submission-ready", "blocked"}:
        raise _error("submission-verdict-route", "submission promotion requires a finalization verdict route")
    outcome = verdict.get("selection_outcome")
    if outcome not in {"improved", "fallback-retained"}:
        raise _error("submission-outcome-invalid", "verdict requires selection_outcome improved|fallback-retained")
    result = dict(state)
    if outcome == "fallback-retained":
        return result
    candidate = verdict.get("final_candidate_path")
    report = verdict.get("final_report_path")
    if not isinstance(candidate, str) or not candidate or not isinstance(report, str) or not report:
        raise _error("submission-partial-pair", "an improved winner requires both final candidate and report paths")
    result["last_accepted_kernel"] = candidate
    result["last_accepted_report"] = report
    return result


def _input_hash(value: Any) -> str:
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if not key.startswith("_")}
    return sha256_canonical_json(value)


def validate_verdict(verdict_path: Path, *, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the attribution verdict against normalized role inputs."""
    verdict = load_json_document(Path(verdict_path), artifact="verdict")
    if verdict.get("schema_version") != 1:
        raise _error("verdict-schema-version", "verdict schema_version must be 1")
    finalization = verdict.get("artifact_kind") == "submission-finalization" or verdict.get("route") in {"submission-ready", "blocked"}
    required_hash_fields = (
        ("decision_sha256", "binding_sha256", "profile_sha256", "report_fact_pack_sha256")
        if finalization
        else ("decision_sha256", "sketch_sha256", "binding_sha256", "profile_sha256", "report_fact_pack_sha256")
    )
    for field in required_hash_fields:
        if not isinstance(verdict.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", verdict[field]):
            raise _error("verdict-hash-invalid", f"verdict requires {field}")
    _validate_verdict_hashes(verdict, inputs)
    facts = inputs.get("facts")
    if not isinstance(facts, dict):
        raise _error("verdict-inputs", "verdict validation requires the fact pack input")

    if finalization:
        return _validate_finalization_verdict(verdict, inputs, facts)
    return _validate_campaign_verdict(verdict, inputs, facts)


def _validate_verdict_hashes(verdict: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    expected = {
        "decision_sha256": inputs.get("decision"),
        "sketch_sha256": inputs.get("sketch"),
        "binding_sha256": inputs.get("binding"),
        "profile_sha256": inputs.get("profile"),
        "report_fact_pack_sha256": inputs.get("facts"),
    }
    for field, input_value in expected.items():
        if input_value is None:
            continue
        if verdict.get(field) != _input_hash(input_value):
            raise _error("verdict-hash-mismatch", f"verdict {field} does not match its input artifact", None)


def _validate_campaign_verdict(verdict: Mapping[str, Any], inputs: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    rule_id = verdict.get("rule_id")
    if rule_id is None:
        classification = verdict.get("classification")
        terminal = verdict.get("terminal_result")
        if classification not in {"none", "not-applicable"} or terminal not in {"accepted", "no-improvement", "screened-out", "aborted"}:
            raise _error("verdict-rule-required", "a rule-less verdict requires classification none and a non-failure terminal result")
        return {"valid": True, "classification": classification, "terminal_result": terminal, "failed_attempt_effect": None, "route": None}

    if rule_id not in RULES:
        raise _error("verdict-rule-unknown", f"unknown rule id {rule_id!r}")
    rule = RULES[rule_id]
    preconditions = verdict.get("preconditions")
    if not isinstance(preconditions, list):
        raise _error("verdict-preconditions", "verdict requires a preconditions list")
    _evaluate_preconditions(rule_id, preconditions, inputs, facts)

    repairable = rule.get("repairable") is True
    route = verdict.get("route")
    repair_exhausted = verdict.get("repair_exhausted") is True
    if repairable:
        if route == "coder-repair" and not repair_exhausted:
            if verdict.get("terminal_result") is not None or verdict.get("failed_attempt_effect") is not None:
                raise _error("verdict-repair-route", "a coder-repair route must not carry a terminal result or counter effect")
            return {
                "valid": True,
                "classification": verdict.get("classification"),
                "terminal_result": None,
                "failed_attempt_effect": None,
                "route": "coder-repair",
            }
        if not repair_exhausted:
            raise _error("verdict-repair-exhausted", "a repairable rule terminal branch requires repair_exhausted true")

    classification = verdict.get("classification")
    terminal = verdict.get("terminal_result")
    effect = verdict.get("failed_attempt_effect")
    if rule_id == "EVIDENCE.OBSERVABLE.MISSING":
        cause = facts.get("evidence_gap_cause")
        if cause == "decision":
            resolved = rule["cause_overrides"]["decision"]
        else:
            resolved = rule
        if classification != resolved["classification"] or terminal != resolved["terminal_result"] or effect != resolved["failed_attempt_effect"]:
            raise _error("verdict-rule-mismatch", "verdict fields do not match the evidence-gap rule resolution")
        return {
            "valid": True,
            "classification": classification,
            "terminal_result": terminal,
            "failed_attempt_effect": effect,
            "route": route,
        }

    if classification != rule["classification"] or terminal != rule["terminal_result"] or effect != rule["failed_attempt_effect"]:
        raise _error("verdict-rule-mismatch", "verdict fields do not match the declared rule")
    return {"valid": True, "classification": classification, "terminal_result": terminal, "failed_attempt_effect": effect, "route": route}


def _evaluate_preconditions(
    rule_id: str,
    preconditions: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> None:
    sketch = inputs.get("sketch") or {}
    binding = inputs.get("binding") or {}
    decision = inputs.get("decision") or {}
    causal_graph = decision.get("causal_graph")
    observable_names = [observable.get("name") for observable in facts.get("observables") or [] if isinstance(observable, dict)]

    def causal_valid() -> bool:
        try:
            validate_causal_graph(
                causal_graph,
                intervention=decision.get("intervention", "") or decision.get("optimization_intent", {}).get("intervention", ""),
                observable_names=observable_names,
            )
            return True
        except VerdictValidationError:
            return False

    evaluators = {
        "sketch_valid": lambda: bool(sketch.get("valid")),
        "causal_graph_valid": causal_valid,
        "binding_valid": lambda: bool(binding.get("valid")),
        "correctness_pass": lambda: (facts.get("correctness") or {}).get("status") == "pass",
        "lowering_observed": lambda: (facts.get("lowering") or {}).get("status") == "observed",
        "expected_mechanism_absent": lambda: (facts.get("lowering") or {}).get("expected_mechanism") == "absent",
        "observable_observed": lambda: all(observable.get("status") == "observed" for observable in facts.get("observables") or []),
        "bounded_probe_attempted": lambda: facts.get("bounded_probe_attempted") is True,
    }
    declared = {item.get("name"): item.get("status") for item in preconditions if isinstance(item, dict)}
    required = RULE_PRECONDITIONS[rule_id]
    missing = [name for name in required if name not in declared]
    if missing:
        raise _error("verdict-preconditions", f"rule {rule_id} requires preconditions {sorted(missing)}")
    for name, status in declared.items():
        if name not in evaluators:
            raise _error("verdict-preconditions", f"unknown precondition {name!r}")
        actual = "pass" if evaluators[name]() else "fail"
        if status != actual:
            raise _error("verdict-precondition-mismatch", f"precondition {name!r} is {actual} but verdict declares {status}")


def _validate_finalization_verdict(verdict: Mapping[str, Any], inputs: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    for field in ("classification", "terminal_result", "failed_attempt_effect", "total_rounds", "performance_miss_streak", "failed_attempt_streak", "workflow_status"):
        if field in verdict:
            raise _error("verdict-finalization-fields", f"finalization verdict must not carry {field!r}")
    route = verdict.get("route")
    if route not in {"submission-ready", "blocked"}:
        raise _error("verdict-finalization-route", "finalization verdict route must be submission-ready|blocked")
    if verdict.get("artifact_kind") != "submission-finalization" or not isinstance(verdict.get("artifact_index"), str):
        raise _error("verdict-finalization-metadata", "finalization verdict requires artifact_kind submission-finalization and artifact_index")

    decision = inputs.get("decision") or {}
    contract = decision.get("final_tuning_contract") or {}
    if verdict.get("submission_snapshot_id") != contract.get("submission_snapshot_id"):
        raise _error("verdict-finalization-snapshot", "verdict submission_snapshot_id must match the Decision contract")

    tuning = facts.get("final_configuration_tuning")
    if not isinstance(tuning, dict):
        raise _error("verdict-finalization-facts", "finalization report requires final_configuration_tuning facts")
    if tuning.get("submission_snapshot_id") != contract.get("submission_snapshot_id"):
        raise _error("verdict-finalization-facts", "final tuning facts submission_snapshot_id must match the Decision")

    selected = select_final_tuning_configuration(contract, tuning.get("search_trials") or [])
    selected_sha = sha256_canonical_json(selected)
    if verdict.get("selected_configuration_sha256") != selected_sha:
        raise _error("verdict-finalization-selection", "verdict selected configuration does not match the deterministic selector")
    final_config_sha = verdict.get("final_configuration_sha256")
    if final_config_sha != sha256_canonical_json(tuning.get("selected_configuration") or {}):
        raise _error("verdict-finalization-selection", "verdict final configuration does not match the report")
    final_candidate_sha = verdict.get("final_candidate_sha256")
    if final_candidate_sha != (tuning.get("post_pin_official") or {}).get("candidate_sha256"):
        raise _error("verdict-finalization-candidate", "verdict final candidate hash must match post-pin official evidence")
    final_binding_sha = verdict.get("final_binding_sha256")
    if final_binding_sha != _input_hash(inputs.get("binding")):
        raise _error("verdict-finalization-binding", "verdict final binding hash must match the validated binding")
    if tuning.get("temporary_storage_clean") is not True:
        raise _error("verdict-finalization-storage", "temporary tuning storage must contain no derived candidate source")

    gates = verdict.get("post_pin_gates")
    if not isinstance(gates, dict):
        raise _error("verdict-finalization-gates", "finalization verdict requires post_pin_gates")
    binding_valid = bool((inputs.get("binding") or {}).get("valid"))
    official = tuning.get("post_pin_official") or {}
    expected_gates = {
        "binding_valid": binding_valid,
        "correctness": (official.get("correctness") or {}).get("status") == "pass" if isinstance(official.get("correctness"), dict) else False,
        "lowering": (official.get("lowering") or {}).get("status") == "observed",
        "promotion_evidence": official.get("promotion_evidence") is True,
        "official_evidence": official.get("official_evidence") is True,
    }
    for name, expected in expected_gates.items():
        if gates.get(name) is not expected:
            raise _error("verdict-finalization-gates", f"post-pin gate {name!r} does not match the report evidence")
    if route == "submission-ready" and not all(gates.get(name) is True for name in expected_gates):
        raise _error("verdict-finalization-gates", "submission-ready requires every post-pin gate to pass")

    selection_outcome = tuning.get("selection_outcome")
    if selection_outcome not in {"improved", "fallback-retained"}:
        raise _error("verdict-finalization-outcome", "final tuning facts require selection_outcome improved|fallback-retained")
    normalized = {
        "valid": True,
        "route": route,
        "submission_snapshot_id": verdict["submission_snapshot_id"],
        "artifact_index": verdict["artifact_index"],
        "selection_outcome": selection_outcome,
        "selected_configuration_sha256": verdict["selected_configuration_sha256"],
        "final_configuration_sha256": final_config_sha,
        "final_candidate_sha256": final_candidate_sha,
        "final_binding_sha256": final_binding_sha,
        "post_pin_gates": gates,
    }
    if selection_outcome == "improved":
        candidate = verdict.get("final_candidate_path")
        report = verdict.get("final_report_path")
        if not isinstance(candidate, str) or not candidate or not isinstance(report, str) or not report:
            raise _error("verdict-finalization-pair", "an improved outcome requires final candidate and report paths")
        normalized["final_candidate_path"] = candidate
        normalized["final_report_path"] = report
    return normalized

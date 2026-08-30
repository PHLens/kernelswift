#!/usr/bin/env python3
"""Validate and normalize a kernel-opt-loop decision artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Sequence

from vnext_common import (
    ContractValidationError,
    compute_submission_snapshot_id,
    load_json_document,
    require_relative_artifact,
    sha256_canonical_json,
    sha256_file,
)
from validate_sketch import SketchValidationError, validate_sketch
from validate_profile import ProfileValidationError, load_profile, validate_configuration_domain


REQUIRED_SECTIONS = (
    "Metadata",
    "Optimization Intent",
    "Unified Sketch",
    "Host Plan",
    "Evaluation Contract",
    "Pitfalls and Anti-pattern Consultation",
    "Rationale and Evidence",
)

SKETCH_HEADERS = (
    ("D", "# D Declarations", ("tensor", "tile", "scalar")),
    ("O", "# O Operations", ("alloc", "load", "compute", "store")),
    ("C", "# C Control", ("parallel", "for", "if", "else", "guard", "end")),
    ("H", "# H Target Hints", ()),
)

METADATA_FIELDS = {
    "schema_version": int,
    "decision": str,
    "round": str,
    "reference_implementation": str,
    "reference_report": str,
    "language": str,
    "backend": str,
    "target_profile": str,
    "runtime_fingerprint_ref": str,
    "change_scope": str,
    "change_family": str,
}

INTENT_FIELDS = {
    "bottleneck_class": str,
    "intervention": str,
    "allowed_changes": list,
    "invariants": list,
    "expected_wall_improvement_pct": (int, float),
}

HOST_PLAN_FIELDS = {
    "affected_scope": list,
    "state_owner": str,
    "lifetime": str,
    "allocation_reuse": str,
    "cache_key": list,
    "invalidation": str,
    "concurrency": str,
    "device_stream_behavior": str,
    "unchanged_behavior": list,
}

ABORT_HOST_PLAN = {"applicability": "not-applicable", "reason": "aborted"}
ABORT_EVALUATION = {"applicability": "not-applicable", "reason": "aborted"}


class DecisionValidationError(ValueError):
    """A stable, line-addressable decision validation failure."""

    def __init__(self, code: str, message: str, line: int = 1) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.line = line

    def __str__(self) -> str:
        return f"{self.line}: {self.code}: {self.message}"


@dataclass(frozen=True)
class Section:
    """An H2 section and the source position of its heading and body."""

    heading: str
    line: int
    body: str
    body_line: int


def extract_sections(text: str) -> dict[str, Section]:
    """Return H2 sections in source order, rejecting duplicate headings."""

    lines = text.splitlines(keepends=True)
    headings: list[tuple[str, int, int]] = []
    heading_pattern = re.compile(r"^##[ \t]+(.+?)[ \t]*(?:\r?\n)?$")
    fence_pattern = re.compile(r"^[ \t]*(`{3,}|~{3,})")
    active_fence: tuple[str, int] | None = None

    for index, line in enumerate(lines):
        fence = fence_pattern.match(line)
        if fence:
            marker = fence.group(1)
            marker_kind = marker[0]
            if active_fence is None:
                active_fence = (marker_kind, len(marker))
            elif marker_kind == active_fence[0] and len(marker) >= active_fence[1]:
                active_fence = None
            continue
        if active_fence is not None:
            continue
        match = heading_pattern.match(line)
        if not match:
            continue
        heading = match.group(1)
        if any(existing[0] == heading for existing in headings):
            raise DecisionValidationError(
                "section-duplicate",
                f"duplicate H2 section {heading!r}",
                index + 1,
            )
        headings.append((heading, index, index + 1))

    sections: dict[str, Section] = {}
    for position, (heading, line_index, body_index) in enumerate(headings):
        body_end = headings[position + 1][1] if position + 1 < len(headings) else len(lines)
        sections[heading] = Section(
            heading=heading,
            line=line_index + 1,
            body="".join(lines[body_index:body_end]),
            body_line=body_index + 1,
        )
    return sections


def parse_single_json_block(section: Section) -> dict[str, Any]:
    """Parse a section containing exactly one fenced JSON object."""

    match = re.fullmatch(
        r"[ \t\r\n]*```json[ \t]*\r?\n(.*?)\r?\n```[ \t]*[\r\n]*",
        section.body,
        flags=re.DOTALL,
    )
    if not match:
        raise DecisionValidationError(
            "json-block-required",
            f"{section.heading} must contain exactly one fenced json object",
            section.line,
        )

    source = match.group(1)
    prefix = section.body[: match.start(1)]
    json_line = section.body_line + prefix.count("\n")
    def reject_nonstandard_constant(constant: str) -> None:
        raise ValueError(f"non-standard numeric constant {constant}")

    try:
        value = json.loads(source, parse_constant=reject_nonstandard_constant)
    except ValueError as error:
        if not isinstance(error, json.JSONDecodeError):
            raise DecisionValidationError(
                "json-invalid",
                f"invalid JSON in {section.heading}: {error}",
                json_line,
            ) from error
        raise DecisionValidationError(
            "json-invalid",
            f"invalid JSON in {section.heading}: {error.msg}",
            json_line + error.lineno - 1,
        ) from error
    if not isinstance(value, dict):
        raise DecisionValidationError(
            "json-object-required",
            f"{section.heading} JSON must be an object",
            json_line,
        )
    return value


def parse_sketch(section: Section, target_profile: str) -> dict[str, list[str]]:
    """Parse the four ordered D/O/C/H sections of one Sketch block."""

    opening_fences = list(re.finditer(r"^```sketch[ \t]*$", section.body, re.MULTILINE))
    if not opening_fences:
        raise DecisionValidationError(
            "sketch-fence-missing",
            "Unified Sketch must contain one fenced sketch block",
            section.line,
        )
    if len(opening_fences) != 1:
        raise DecisionValidationError(
            "sketch-fence-count",
            "Unified Sketch must contain exactly one fenced sketch block",
            section.line,
        )

    match = re.fullmatch(
        r"[ \t\r\n]*```sketch[ \t]*\r?\n(.*?)\r?\n```[ \t]*[\r\n]*",
        section.body,
        flags=re.DOTALL,
    )
    if not match:
        raise DecisionValidationError(
            "sketch-fence-count",
            "Unified Sketch must contain only one complete fenced sketch block",
            section.line,
        )

    content = match.group(1)
    prefix = section.body[: match.start(1)]
    first_content_line = section.body_line + prefix.count("\n")
    content_lines = content.splitlines()
    header_positions: list[int] = []

    for _key, expected_header, _prefixes in SKETCH_HEADERS:
        positions = [index for index, line in enumerate(content_lines) if line.strip() == expected_header]
        if len(positions) != 1:
            raise DecisionValidationError(
                "sketch-header-required",
                f"Sketch requires exactly one {expected_header!r} header",
                first_content_line,
            )
        header_positions.append(positions[0])

    if header_positions != sorted(header_positions) or header_positions[0] != 0:
        raise DecisionValidationError(
            "sketch-header-order",
            "Sketch headers must appear once in D, O, C, H order",
            first_content_line,
        )

    parsed: dict[str, list[str]] = {}
    for index, (key, _header, allowed_prefixes) in enumerate(SKETCH_HEADERS):
        start = header_positions[index] + 1
        end = header_positions[index + 1] if index + 1 < len(SKETCH_HEADERS) else len(content_lines)
        statements: list[str] = []
        for offset in range(start, end):
            raw_statement = content_lines[offset].strip()
            if not raw_statement:
                continue
            statement = " ".join(raw_statement.split())
            source_line = first_content_line + offset
            if key in {"D", "O", "C"}:
                first_word = statement.split(maxsplit=1)[0]
                if first_word not in allowed_prefixes:
                    raise DecisionValidationError(
                        f"sketch-{key.lower()}-statement-invalid",
                        f"{key} statement must start with {'|'.join(allowed_prefixes)}",
                        source_line,
                    )
            statements.append(statement)
        if not statements:
            raise DecisionValidationError(
                f"sketch-{key.lower()}-section-empty",
                f"Sketch section {key} must contain at least one statement",
                first_content_line + start,
            )
        parsed[key] = statements

    expected_target = f"target={target_profile}"
    if parsed["H"][0] != expected_target:
        raise DecisionValidationError(
            "sketch-target-mismatch",
            f"first H directive must be exactly {expected_target!r}",
            first_content_line + header_positions[3] + 1,
        )
    directive_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*=[^\s=]+$")
    for index, directive in enumerate(parsed["H"]):
        if not directive_pattern.fullmatch(directive):
            raise DecisionValidationError(
                "sketch-h-one-directive-per-line",
                "each H line must contain exactly one name=value directive",
                first_content_line + header_positions[3] + 1 + index,
            )
    return parsed


def _require_fields(
    value: dict[str, Any],
    fields: dict[str, type | tuple[type, ...]],
    *,
    code_prefix: str,
    line: int,
) -> None:
    for field, expected_type in fields.items():
        if field not in value:
            raise DecisionValidationError(
                f"{code_prefix}-field-required",
                f"missing required field {field!r}",
                line,
            )
        actual = value[field]
        if isinstance(actual, bool) or not isinstance(actual, expected_type):
            if isinstance(expected_type, tuple):
                type_name = " or ".join(item.__name__ for item in expected_type)
            else:
                type_name = expected_type.__name__
            raise DecisionValidationError(
                f"{code_prefix}-field-type",
                f"field {field!r} must be {type_name}",
                line,
            )


def _require_nonempty_string(value: Any, field: str, code_prefix: str, line: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DecisionValidationError(
            f"{code_prefix}-field-empty",
            f"field {field!r} must be a nonempty string",
            line,
        )


def _require_string_list(
    value: Any,
    field: str,
    code_prefix: str,
    line: int,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise DecisionValidationError(
            f"{code_prefix}-field-empty",
            f"field {field!r} must be a {'possibly empty ' if allow_empty else 'nonempty '}list of strings",
            line,
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise DecisionValidationError(
            f"{code_prefix}-field-type",
            f"field {field!r} must contain only nonempty strings",
            line,
        )


def _validate_reference(value: str, field: str, line: int, *, require_anchor: bool = False) -> None:
    path_text, separator, anchor = value.partition("#")
    path = PurePosixPath(path_text)
    invalid = not path_text or path.is_absolute() or ".." in path.parts
    if require_anchor:
        invalid = invalid or not separator or not anchor
    elif separator:
        invalid = invalid or not anchor
    if invalid:
        raise DecisionValidationError(
            "metadata-reference-invalid",
            f"field {field!r} must be a relative artifact reference"
            + (" with an anchor" if require_anchor else ""),
            line,
        )


def _validate_metadata(
    metadata: dict[str, Any],
    section: Section,
    expected_profile: str | None,
) -> None:
    _require_fields(metadata, METADATA_FIELDS, code_prefix="metadata", line=section.line)
    if metadata["schema_version"] != 1:
        raise DecisionValidationError(
            "schema-version-unsupported",
            "schema_version must be 1",
            section.line,
        )
    if metadata["decision"] not in {"proceed", "abort"}:
        raise DecisionValidationError(
            "decision-enum-invalid",
            "decision must be proceed or abort",
            section.line,
        )
    if metadata["change_scope"] not in {"kernel", "host", "mixed", "none"}:
        raise DecisionValidationError(
            "change-scope-enum-invalid",
            "change_scope must be kernel, host, mixed, or none",
            section.line,
        )
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", metadata["change_family"]):
        raise DecisionValidationError(
            "metadata-change-family-invalid",
            "change_family must be a lowercase hyphen-separated slug",
            section.line,
        )
    if not re.fullmatch(r"[0-9]{3}", metadata["round"]):
        raise DecisionValidationError(
            "round-format-invalid",
            "round must contain exactly three decimal digits",
            section.line,
        )
    for field in (
        "reference_implementation",
        "reference_report",
        "language",
        "backend",
        "target_profile",
        "runtime_fingerprint_ref",
    ):
        _require_nonempty_string(metadata[field], field, "metadata", section.line)
    _validate_reference(metadata["reference_implementation"], "reference_implementation", section.line)
    _validate_reference(metadata["reference_report"], "reference_report", section.line)
    _validate_reference(
        metadata["runtime_fingerprint_ref"],
        "runtime_fingerprint_ref",
        section.line,
        require_anchor=True,
    )
    if expected_profile is not None and metadata["target_profile"] != expected_profile:
        raise DecisionValidationError(
            "target-profile-mismatch",
            f"target_profile {metadata['target_profile']!r} does not match expected profile {expected_profile!r}",
            section.line,
        )
    expected_scope = "none" if metadata["decision"] == "abort" else None
    if expected_scope is not None and metadata["change_scope"] != expected_scope:
        raise DecisionValidationError(
            "abort-scope-invalid",
            "an abort decision must use change_scope none",
            section.line,
        )
    if metadata["decision"] == "proceed" and metadata["change_scope"] == "none":
        raise DecisionValidationError(
            "proceed-scope-invalid",
            "a proceeding decision must use kernel, host, or mixed change_scope",
            section.line,
        )


def _validate_intent(intent: dict[str, Any], section: Section, *, aborted: bool) -> None:
    _require_fields(intent, INTENT_FIELDS, code_prefix="intent", line=section.line)
    for field in ("bottleneck_class", "intervention"):
        _require_nonempty_string(intent[field], field, "intent", section.line)
    _require_string_list(
        intent["allowed_changes"],
        "allowed_changes",
        "intent",
        section.line,
        allow_empty=aborted,
    )
    _require_string_list(intent["invariants"], "invariants", "intent", section.line)
    if intent["expected_wall_improvement_pct"] < 0:
        raise DecisionValidationError(
            "intent-improvement-invalid",
            "expected_wall_improvement_pct must not be negative",
            section.line,
        )


def _validate_required_host_plan(host_plan: dict[str, Any], section: Section) -> None:
    if host_plan.get("applicability") != "required":
        raise DecisionValidationError(
            "host-plan-required",
            "host and mixed changes require an applicable Host Plan",
            section.line,
        )
    _require_fields(host_plan, HOST_PLAN_FIELDS, code_prefix="host-plan", line=section.line)
    for field in (
        "state_owner",
        "lifetime",
        "allocation_reuse",
        "invalidation",
        "concurrency",
        "device_stream_behavior",
    ):
        _require_nonempty_string(host_plan[field], field, "host-plan", section.line)
    for field in ("affected_scope", "cache_key", "unchanged_behavior"):
        _require_string_list(host_plan[field], field, "host-plan", section.line)


def _validate_evaluation(
    evaluation: dict[str, Any],
    section: Section,
    intent: dict[str, Any],
    round_number: str,
) -> None:
    required = {
        "hypothesis_id": str,
        "intervention": str,
        "expected_causal_chain": list,
        "primary_metric": dict,
        "mechanism_observables": list,
        "guardrails": list,
        "profiling_level": str,
    }
    _require_fields(evaluation, required, code_prefix="evaluation", line=section.line)
    if evaluation["hypothesis_id"] != f"H-{round_number}":
        raise DecisionValidationError(
            "evaluation-hypothesis-id-invalid",
            f"hypothesis_id must be H-{round_number}",
            section.line,
        )
    _require_nonempty_string(evaluation["intervention"], "intervention", "evaluation", section.line)
    if evaluation["intervention"] != intent["intervention"]:
        raise DecisionValidationError(
            "evaluation-intervention-mismatch",
            "Evaluation Contract intervention must match Optimization Intent",
            section.line,
        )
    _require_string_list(
        evaluation["expected_causal_chain"],
        "expected_causal_chain",
        "evaluation",
        section.line,
    )

    primary_metric = evaluation["primary_metric"]
    if primary_metric.get("name") != "wall_time":
        raise DecisionValidationError(
            "evaluation-primary-metric-invalid",
            "primary_metric.name must be wall_time",
            section.line,
        )
    threshold = primary_metric.get("expected_improvement_pct")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or float(threshold) != 5.0:
        raise DecisionValidationError(
            "evaluation-threshold-invalid",
            "primary_metric.expected_improvement_pct must be 5.0",
            section.line,
        )

    observables = evaluation["mechanism_observables"]
    if not observables:
        raise DecisionValidationError(
            "evaluation-observable-required",
            "at least one mechanism observable is required",
            section.line,
        )
    if not isinstance(observables, list):
        raise DecisionValidationError(
            "evaluation-field-type",
            "mechanism_observables must be a list",
            section.line,
        )
    for observable in observables:
        if not isinstance(observable, dict):
            raise DecisionValidationError(
                "evaluation-observable-invalid",
                "each mechanism observable must be an object",
                section.line,
            )
        for field in ("name", "expectation"):
            if not isinstance(observable.get(field), str) or not observable[field].strip():
                raise DecisionValidationError(
                    "evaluation-observable-invalid",
                    f"each mechanism observable requires nonempty {field}",
                    section.line,
                )

    _require_string_list(
        evaluation["guardrails"],
        "guardrails",
        "evaluation",
        section.line,
    )
    if "correctness:pass" not in evaluation["guardrails"]:
        raise DecisionValidationError(
            "evaluation-correctness-guardrail-required",
            "guardrails must include correctness:pass",
            section.line,
        )
    if evaluation["profiling_level"] not in {"summary", "targeted", "deep-on-demand"}:
        raise DecisionValidationError(
            "evaluation-profiling-level-invalid",
            "profiling_level must be summary, targeted, or deep-on-demand",
            section.line,
        )


def _validate_title(text: str, round_number: str) -> None:
    lines = text.splitlines()
    first_nonblank = next(((index, line) for index, line in enumerate(lines) if line.strip()), None)
    if first_nonblank is None or first_nonblank[1].strip() != f"# Decision {round_number}":
        line = first_nonblank[0] + 1 if first_nonblank else 1
        raise DecisionValidationError(
            "decision-title-invalid",
            f"document must begin with # Decision {round_number}",
            line,
        )


def validate_decision(
    path: Path,
    expected_profile: str | None = None,
    *,
    project_root: Path | None = None,
    expected_implementation_profile: str | None = None,
) -> dict[str, object]:
    """Validate *path* and return its normalized machine-readable contract.

    Schema-v1 keeps the legacy D/O/C/H parser and ``expected_profile``.
    Schema-v2 requires ``expected_implementation_profile`` (and optional
    ``project_root``) and validates the typed Sketch, frozen implementation
    profile snapshot, project capability claim, causal graph, fallback
    provenance, and finite final-tuning contract.
    """

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DecisionValidationError("decision-read-error", str(error), 1) from error

    sections = extract_sections(text)
    for heading in REQUIRED_SECTIONS:
        if heading not in sections:
            raise DecisionValidationError(
                "section-missing",
                f"missing required H2 section {heading!r}",
                1,
            )

    metadata = parse_single_json_block(sections["Metadata"])
    schema_version = metadata.get("schema_version")
    if schema_version == 1:
        if expected_implementation_profile is not None:
            raise DecisionValidationError(
                "implementation-profile-v1-invalid",
                "schema-v1 uses expected_profile",
                sections["Metadata"].line,
            )
        unknown_sections = [heading for heading in sections if heading not in REQUIRED_SECTIONS]
        if unknown_sections:
            first = sections[unknown_sections[0]]
            raise DecisionValidationError(
                "section-unknown",
                f"unknown H2 section {first.heading!r}",
                first.line,
            )
        return _validate_v1_decision(text, sections, metadata, expected_profile)
    if schema_version == 2:
        if expected_profile is not None or expected_implementation_profile is None:
            raise DecisionValidationError(
                "implementation-profile-v2-required",
                "schema-v2 requires expected_implementation_profile only",
                sections["Metadata"].line,
            )
        root = _resolve_project_root(path, project_root)
        _validate_v2_section_set(sections, metadata)
        _validate_metadata_v2(metadata, sections["Metadata"], path)
        _validate_title_v2(text, metadata)
        return _validate_vnext_decision(path, root, sections, metadata, expected_implementation_profile)
    raise DecisionValidationError(
        "metadata-schema-version",
        "schema_version must be 1 or 2",
        sections["Metadata"].line,
    )


def _validate_v1_decision(
    text: str,
    sections: dict[str, Section],
    metadata: dict[str, Any],
    expected_profile: str | None,
) -> dict[str, object]:
    _validate_metadata(metadata, sections["Metadata"], expected_profile)
    _validate_title(text, metadata["round"])

    aborted = metadata["decision"] == "abort"
    intent = parse_single_json_block(sections["Optimization Intent"])
    _validate_intent(intent, sections["Optimization Intent"], aborted=aborted)

    sketch_section = sections["Unified Sketch"]
    host_section = sections["Host Plan"]
    evaluation_section = sections["Evaluation Contract"]
    host_plan = parse_single_json_block(host_section)

    if aborted:
        if sketch_section.body.strip() != "N/A: aborted":
            raise DecisionValidationError(
                "abort-sketch-marker-invalid",
                "an abort decision must use the exact Unified Sketch marker 'N/A: aborted'",
                sketch_section.line,
            )
        if host_plan != ABORT_HOST_PLAN:
            raise DecisionValidationError(
                "abort-host-plan-invalid",
                "an abort decision requires the not-applicable aborted Host Plan",
                host_section.line,
            )
        evaluation = parse_single_json_block(evaluation_section)
        if evaluation != ABORT_EVALUATION:
            raise DecisionValidationError(
                "abort-evaluation-invalid",
                "an abort decision requires the not-applicable aborted Evaluation Contract",
                evaluation_section.line,
            )
        sketch = None
    else:
        scope = metadata["change_scope"]
        if scope == "host":
            if sketch_section.body.strip() != "N/A: host-only change":
                raise DecisionValidationError(
                    "host-sketch-marker-invalid",
                    "a host-only decision must use the exact Unified Sketch marker 'N/A: host-only change'",
                    sketch_section.line,
                )
            sketch = None
        else:
            sketch = parse_sketch(sketch_section, metadata["target_profile"])

        if scope in {"host", "mixed"}:
            _validate_required_host_plan(host_plan, host_section)
        else:
            if host_plan.get("applicability") != "not-applicable":
                raise DecisionValidationError(
                    "kernel-host-plan-invalid",
                    "a kernel-only decision requires a not-applicable Host Plan",
                    host_section.line,
                )
            _require_nonempty_string(host_plan.get("reason"), "reason", "host-plan", host_section.line)

        evaluation = parse_single_json_block(evaluation_section)
        _validate_evaluation(evaluation, evaluation_section, intent, metadata["round"])

    for heading in ("Pitfalls and Anti-pattern Consultation", "Rationale and Evidence"):
        if not sections[heading].body.strip():
            raise DecisionValidationError(
                "section-content-required",
                f"{heading} must not be empty",
                sections[heading].line,
            )

    return {
        "valid": True,
        "metadata": metadata,
        "optimization_intent": intent,
        "sketch": sketch,
        "host_plan": host_plan,
        "evaluation_contract": evaluation,
    }


def _resolve_project_root(decision: Path, project_root: Path | None) -> Path:
    if project_root is not None:
        root = Path(project_root).resolve()
    else:
        root = decision.resolve().parents[1]
    if not root.is_dir():
        raise DecisionValidationError("project-root-invalid", f"project root {root} is not a directory", 1)
    return root


def _require_v2_artifact(root: Path, reference: str, section: Section) -> Path:
    try:
        return require_relative_artifact(root, reference)
    except ContractValidationError as error:
        raise DecisionValidationError("artifact-missing", error.message, section.line) from error


def _validate_v2_section_set(sections: dict[str, Section], metadata: dict[str, Any]) -> None:
    decision_kind = metadata.get("decision_kind")
    for heading in sections:
        if heading in REQUIRED_SECTIONS:
            continue
        if heading == "Final Configuration Tuning" and decision_kind == "final-autotune":
            continue
        section = sections[heading]
        raise DecisionValidationError("section-unknown", f"unknown H2 section {heading!r}", section.line)
    if decision_kind == "final-autotune" and "Final Configuration Tuning" not in sections:
        raise DecisionValidationError(
            "final-tuning-section-required",
            "a final-autotune decision requires the Final Configuration Tuning section",
            1,
        )
    if decision_kind == "optimization" and "Final Configuration Tuning" in sections:
        raise DecisionValidationError(
            "final-tuning-section-invalid",
            "an optimization decision must not contain the Final Configuration Tuning section",
            sections["Final Configuration Tuning"].line,
        )


def _validate_metadata_v2(metadata: dict[str, Any], section: Section, path: Path) -> None:
    if metadata.get("decision") != "proceed":
        raise DecisionValidationError("decision-enum-invalid", "schema-v2 requires decision proceed", section.line)
    decision_kind = metadata.get("decision_kind")
    if decision_kind not in {"optimization", "final-autotune"}:
        raise DecisionValidationError("decision-kind-invalid", "decision_kind must be optimization or final-autotune", section.line)
    if decision_kind == "optimization":
        if "artifact_index" in metadata:
            raise DecisionValidationError("artifact-index-invalid", "optimization decisions use round, not artifact_index", section.line)
        if not isinstance(metadata.get("round"), str) or not re.fullmatch(r"[0-9]{3}", metadata["round"]):
            raise DecisionValidationError("round-format-invalid", "round must contain exactly three decimal digits", section.line)
    else:
        if "round" in metadata:
            raise DecisionValidationError("round-invalid", "final-autotune decisions must not carry a campaign round", section.line)
        artifact_index = metadata.get("artifact_index")
        if not isinstance(artifact_index, str) or not re.fullmatch(r"[0-9]{3}", artifact_index):
            raise DecisionValidationError("artifact-index-invalid", "artifact_index must contain exactly three decimal digits", section.line)
        expected_name = f"decision_{artifact_index}.md"
        if path.name != expected_name:
            raise DecisionValidationError("artifact-index-invalid", f"artifact_index must match the filename {expected_name!r}", section.line)

    for field in ("reference_implementation", "reference_report", "language", "backend", "runtime_fingerprint_ref"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise DecisionValidationError("metadata-field-required", f"missing required field {field!r}", section.line)
    if metadata["change_scope"] not in {"kernel", "host", "mixed", "none"}:
        raise DecisionValidationError("change-scope-enum-invalid", "change_scope must be kernel, host, mixed, or none", section.line)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", metadata.get("change_family", "")):
        raise DecisionValidationError("metadata-change-family-invalid", "change_family must be a lowercase hyphen-separated slug", section.line)
    for field in ("sketch_ref", "implementation_profile_snapshot_ref", "project_capability_claim_ref"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise DecisionValidationError("metadata-reference-invalid", f"field {field!r} must be a relative artifact reference", section.line)
    for field in ("sketch_sha256", "implementation_profile_snapshot_sha256", "project_capability_claim_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise DecisionValidationError("metadata-hash-invalid", f"field {field!r} must be a SHA-256 hex digest", section.line)


def _validate_title_v2(text: str, metadata: dict[str, Any]) -> None:
    identifier = metadata.get("artifact_index") if metadata.get("decision_kind") == "final-autotune" else metadata.get("round")
    lines = text.splitlines()
    first_nonblank = next(((index, line) for index, line in enumerate(lines) if line.strip()), None)
    if first_nonblank is None or first_nonblank[1].strip() != f"# Decision {identifier}":
        line = first_nonblank[0] + 1 if first_nonblank else 1
        raise DecisionValidationError(
            "decision-title-invalid",
            f"document must begin with # Decision {identifier}",
            line,
        )


def _validate_vnext_decision(
    path: Path,
    root: Path,
    sections: dict[str, Section],
    metadata: dict[str, Any],
    expected_implementation_profile: str,
) -> dict[str, object]:
    sketch_result = _validate_v2_sketch(sections, metadata, root)
    profile = _validate_v2_profile_snapshot(sections, metadata, root, expected_implementation_profile)
    claim = _validate_v2_claim(sections, metadata, root, expected_implementation_profile)

    intent = parse_single_json_block(sections["Optimization Intent"])
    host_plan = parse_single_json_block(sections["Host Plan"])
    evaluation = parse_single_json_block(sections["Evaluation Contract"])

    fallback_provenance: dict[str, Any] | None = None
    final_tuning_contract: dict[str, Any] | None = None
    if metadata["decision_kind"] == "optimization":
        _validate_intent(intent, sections["Optimization Intent"], aborted=False)
        _validate_v2_host_plan(host_plan, sections["Host Plan"], metadata["change_scope"])
        causal_graph = _validate_v2_evaluation(evaluation, sections["Evaluation Contract"])
        _validate_causal_connectivity(causal_graph, sketch_result, sections["Evaluation Contract"])
        fallback_provenance = _validate_fallback_provenance(intent, claim, sections["Optimization Intent"])
    else:
        final_tuning_contract = _validate_final_tuning_contract(
            sections, metadata, root, profile, sketch_result, claim
        )

    for heading in ("Pitfalls and Anti-pattern Consultation", "Rationale and Evidence"):
        if not sections[heading].body.strip():
            raise DecisionValidationError(
                "section-content-required",
                f"{heading} must not be empty",
                sections[heading].line,
            )

    result: dict[str, object] = {
        "valid": True,
        "metadata": metadata,
        "optimization_intent": intent,
        "sketch": sketch_result,
        "sketch_ref": metadata["sketch_ref"],
        "sketch_sha256": metadata["sketch_sha256"],
        "implementation_profile_snapshot_ref": metadata["implementation_profile_snapshot_ref"],
        "implementation_profile_snapshot_sha256": metadata["implementation_profile_snapshot_sha256"],
        "project_capability_claim_ref": metadata["project_capability_claim_ref"],
        "project_capability_claim_sha256": metadata["project_capability_claim_sha256"],
        "decision_kind": metadata["decision_kind"],
        "host_plan": host_plan,
        "evaluation_contract": evaluation,
        "fallback_provenance": fallback_provenance,
        "final_tuning_contract": final_tuning_contract,
        "causal_graph": evaluation.get("causal_graph"),
    }
    return result


def _validate_v2_sketch(
    sections: dict[str, Section],
    metadata: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    section = sections["Unified Sketch"]
    contract = parse_single_json_block(section)
    if contract.get("artifact") != metadata["sketch_ref"] or contract.get("sha256") != metadata["sketch_sha256"]:
        raise DecisionValidationError(
            "sketch-contract-mismatch",
            "Unified Sketch artifact/sha256 must match Metadata",
            section.line,
        )
    sketch_path = _require_v2_artifact(root, metadata["sketch_ref"], section)
    if sha256_file(sketch_path) != metadata["sketch_sha256"]:
        raise DecisionValidationError("sketch-hash-mismatch", "referenced sketch hash does not match", section.line)
    expected_round = metadata.get("round") if metadata.get("decision_kind") == "optimization" else None
    try:
        return validate_sketch(sketch_path, expected_round=expected_round)
    except SketchValidationError as error:
        raise DecisionValidationError("sketch-invalid", f"typed sketch is invalid: {error.message}", section.line) from error


def _validate_v2_profile_snapshot(
    sections: dict[str, Section],
    metadata: dict[str, Any],
    root: Path,
    expected_implementation_profile: str,
) -> dict[str, Any]:
    section = sections["Metadata"]
    snapshot_path = _require_v2_artifact(root, metadata["implementation_profile_snapshot_ref"], section)
    if sha256_file(snapshot_path) != metadata["implementation_profile_snapshot_sha256"]:
        raise DecisionValidationError("profile-snapshot-hash-mismatch", "implementation profile snapshot hash does not match", section.line)
    try:
        profile = load_profile(snapshot_path)
    except ProfileValidationError as error:
        raise DecisionValidationError("profile-snapshot-invalid", f"implementation profile snapshot is invalid: {error.message}", section.line) from error
    if profile["implementation_profile_id"] != expected_implementation_profile:
        raise DecisionValidationError(
            "implementation-profile-mismatch",
            f"snapshot profile {profile['implementation_profile_id']!r} does not match expected {expected_implementation_profile!r}",
            section.line,
        )
    return profile


def _validate_v2_claim(
    sections: dict[str, Section],
    metadata: dict[str, Any],
    root: Path,
    expected_implementation_profile: str,
) -> dict[str, Any]:
    section = sections["Metadata"]
    claim_path = _require_v2_artifact(root, metadata["project_capability_claim_ref"], section)
    if sha256_file(claim_path) != metadata["project_capability_claim_sha256"]:
        raise DecisionValidationError("claim-hash-mismatch", "project capability claim hash does not match", section.line)
    claim = load_json_document(claim_path, artifact="project capability claim")
    if claim.get("implementation_profile_id") != expected_implementation_profile:
        raise DecisionValidationError("claim-profile-mismatch", "project capability claim profile does not match", section.line)
    _require_runtime_fingerprint_anchor(root, section)
    return claim


def _require_runtime_fingerprint_anchor(root: Path, section: Section) -> None:
    project_md = root / "project.md"
    if not project_md.is_file():
        raise DecisionValidationError("runtime-fingerprint-anchor", "project.md must exist", section.line)
    heading_pattern = re.compile(r"^##[ \t]+(.+?)[ \t]*$")
    headings = []
    for line in project_md.read_text(encoding="utf-8").splitlines():
        match = heading_pattern.match(line)
        if match:
            headings.append(match.group(1).lower().replace("-", " ").strip())
    if "runtime fingerprint" not in headings:
        raise DecisionValidationError("runtime-fingerprint-anchor", "project.md must contain a ## Runtime Fingerprint heading", section.line)


def _validate_v2_host_plan(host_plan: dict[str, Any], section: Section, change_scope: str) -> None:
    if change_scope in {"host", "mixed"}:
        _validate_required_host_plan(host_plan, section)
    else:
        if host_plan.get("applicability") != "not-applicable":
            raise DecisionValidationError(
                "kernel-host-plan-invalid",
                "a kernel-only decision requires a not-applicable Host Plan",
                section.line,
            )
        _require_nonempty_string(host_plan.get("reason"), "reason", "host-plan", section.line)


def _validate_v2_evaluation(evaluation: dict[str, Any], section: Section) -> dict[str, Any]:
    causal_graph = evaluation.get("causal_graph")
    if not isinstance(causal_graph, dict):
        raise DecisionValidationError("causal-graph-required", "Evaluation Contract requires a causal_graph object", section.line)
    nodes = causal_graph.get("nodes")
    edges = causal_graph.get("edges")
    if not isinstance(nodes, list) or not nodes or any(not isinstance(node, str) or not node for node in nodes):
        raise DecisionValidationError("causal-graph-invalid", "causal_graph.nodes must be a nonempty string list", section.line)
    if len(set(nodes)) != len(nodes):
        raise DecisionValidationError("causal-graph-invalid", "causal_graph.nodes must be unique", section.line)
    if not isinstance(edges, list):
        raise DecisionValidationError("causal-graph-invalid", "causal_graph.edges must be a list", section.line)
    node_set = set(nodes)
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or edge[0] not in node_set or edge[1] not in node_set:
            raise DecisionValidationError("causal-graph-invalid", "each causal edge must name two existing nodes", section.line)
    return causal_graph


def _validate_causal_connectivity(
    causal_graph: dict[str, Any],
    sketch_result: dict[str, Any],
    section: Section,
) -> None:
    graph_nodes = set(causal_graph.get("nodes") or [])
    sketch_nodes = set(sketch_result.get("causal_node_ids") or [])
    if sketch_nodes and not sketch_nodes.issubset(graph_nodes):
        missing = sorted(sketch_nodes - graph_nodes)
        raise DecisionValidationError(
            "causal-graph-invalid",
            f"Sketch causal nodes {missing} are not connected to the Evaluation Contract graph",
            section.line,
        )


def _validate_fallback_provenance(
    intent: dict[str, Any],
    claim: dict[str, Any],
    section: Section,
) -> dict[str, Any] | None:
    provenance = intent.get("fallback_provenance")
    uses_substitution = intent.get("uses_algorithm_substitution") is True
    if uses_substitution and provenance is None:
        raise DecisionValidationError(
            "fallback-provenance-required",
            "an algorithm substitution requires explicit fallback provenance",
            section.line,
        )
    if provenance is None:
        return None
    if not isinstance(provenance, dict):
        raise DecisionValidationError("fallback-provenance-invalid", "fallback_provenance must be an object", section.line)
    for field in ("fallback_from", "fallback_kind", "probe_policy"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise DecisionValidationError("fallback-provenance-invalid", f"fallback_provenance requires nonempty {field}", section.line)
    if provenance["fallback_kind"] != "algorithm-substitution":
        raise DecisionValidationError("fallback-provenance-invalid", "fallback_kind must be algorithm-substitution", section.line)
    if provenance["probe_policy"] not in {"optional", "before-fallback", "must-resolve"}:
        raise DecisionValidationError("fallback-provenance-invalid", "probe_policy must be optional|before-fallback|must-resolve", section.line)
    for field in ("primary_signature", "fallback_signature"):
        if not isinstance(provenance.get(field), dict):
            raise DecisionValidationError("fallback-provenance-invalid", f"fallback_provenance requires {field} object", section.line)
    disposition_id = provenance.get("qualification_disposition_id")
    disposition_sha = provenance.get("qualification_disposition_sha256")
    if not isinstance(disposition_id, str) or not disposition_id:
        raise DecisionValidationError("fallback-provenance-invalid", "fallback_provenance requires qualification_disposition_id", section.line)
    if not isinstance(disposition_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", disposition_sha):
        raise DecisionValidationError("fallback-provenance-invalid", "fallback_provenance requires qualification_disposition_sha256", section.line)
    if provenance.get("primary_remains_unknown") is not True:
        raise DecisionValidationError("fallback-provenance-invalid", "fallback provenance requires primary_remains_unknown true", section.line)
    dispositions = {item.get("disposition_id"): item for item in claim.get("qualification_dispositions") or []}
    disposition = dispositions.get(disposition_id)
    if disposition is None:
        raise DecisionValidationError("fallback-disposition-missing", f"disposition {disposition_id!r} is not embedded in the project claim", section.line)
    if sha256_canonical_json(disposition) != disposition_sha:
        raise DecisionValidationError("fallback-disposition-hash", "qualification_disposition_sha256 does not match the embedded disposition", section.line)
    if disposition.get("fallback_authorized") is not True or disposition.get("primary_remains_unknown") is not True:
        raise DecisionValidationError("fallback-disposition-unresolved", "the embedded disposition must be authorized with the primary remaining unknown", section.line)
    return {**provenance, "qualification_disposition_id": disposition_id}


def _validate_final_tuning_contract(
    sections: dict[str, Section],
    metadata: dict[str, Any],
    root: Path,
    profile: dict[str, Any],
    sketch_result: dict[str, Any],
    claim: dict[str, Any],
) -> dict[str, Any]:
    section = sections["Final Configuration Tuning"]
    tuning = parse_single_json_block(section)
    submission_snapshot_id = tuning.get("submission_snapshot_id")
    if not isinstance(submission_snapshot_id, str) or not re.fullmatch(r"[0-9a-f]{64}", submission_snapshot_id):
        raise DecisionValidationError("final-tuning-snapshot-id", "final tuning requires submission_snapshot_id", section.line)

    anchors = tuning.get("anchors")
    if not isinstance(anchors, dict):
        raise DecisionValidationError("final-tuning-anchors", "final tuning requires an anchors object", section.line)
    anchor_keys = {
        "accepted_candidate": "candidate_sha256",
        "accepted_binding": "binding_sha256",
        "sketch": "sketch_sha256",
        "profile": "profile_sha256",
        "claim": "claim_sha256",
        "runtime_snapshot": "runtime_snapshot_sha256",
        "measurement_fingerprint": "measurement_fingerprint_sha256",
        "harness": "harness_sha256",
        "base": "base_sha256",
    }
    if set(anchors) != set(anchor_keys):
        raise DecisionValidationError("final-tuning-anchors", "anchors must name exactly the nine immutable inputs", section.line)
    hashes: dict[str, str] = {}
    for key, canonical_key in anchor_keys.items():
        entry = anchors[key]
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise DecisionValidationError("final-tuning-anchors", f"anchor {key} requires a SHA-256 hash", section.line)
        hashes[canonical_key] = entry["sha256"]
        if key in {"measurement_fingerprint"}:
            continue
        reference = entry.get("ref")
        if not isinstance(reference, str) or not reference:
            raise DecisionValidationError("final-tuning-anchors", f"anchor {key} requires a relative ref", section.line)
        artifact = _require_v2_artifact(root, reference, section)
        if sha256_file(artifact) != entry["sha256"]:
            raise DecisionValidationError("final-tuning-anchor-hash", f"anchor {key} hash does not match its referenced file", section.line)

    if tuning["anchors"]["sketch"]["sha256"] != metadata["sketch_sha256"]:
        raise DecisionValidationError("final-tuning-anchors", "final tuning must reuse the accepted Sketch", section.line)
    computed = compute_submission_snapshot_id(hashes)
    if computed != submission_snapshot_id:
        raise DecisionValidationError("final-tuning-snapshot-id", "submission_snapshot_id does not match the immutable anchors", section.line)

    configurations = tuning.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise DecisionValidationError("final-tuning-domain", "final tuning requires a finite nonempty configurations list", section.line)
    seen_configs: set[str] = set()
    field_names: list[str] = []
    for configuration in configurations:
        if not isinstance(configuration, dict) or not configuration:
            raise DecisionValidationError("final-tuning-domain", "each configuration must be a nonempty object", section.line)
        key = json.dumps(configuration, sort_keys=True)
        if key in seen_configs:
            raise DecisionValidationError("final-tuning-duplicate", "duplicate configuration in the domain", section.line)
        seen_configs.add(key)
        for name in configuration:
            if name not in field_names:
                field_names.append(name)

    tunable = set(sketch_result.get("preferred_hints") or []) | set(sketch_result.get("exploratory_hints") or [])
    for name in field_names:
        if name not in tunable:
            raise DecisionValidationError(
                "final-tuning-semantic-field",
                f"tuning field {name!r} is not a preferred|exploratory configuration-only Sketch hint",
                section.line,
            )

    fallback_configuration = tuning.get("fallback_configuration")
    if not isinstance(fallback_configuration, dict):
        raise DecisionValidationError("final-tuning-domain", "final tuning requires the accepted fallback_configuration", section.line)
    if json.dumps(fallback_configuration, sort_keys=True) not in seen_configs:
        raise DecisionValidationError("final-tuning-domain", "the accepted fallback/control configuration must be in the domain", section.line)

    declared_field_set = set(field_names)
    for configuration in configurations:
        if set(configuration) != declared_field_set:
            raise DecisionValidationError(
                "final-tuning-domain",
                "all configurations must share exactly the declared tuning field set",
                section.line,
            )

    fields = [
        {
            "name": name,
            "values": list(dict.fromkeys(configuration[name] for configuration in configurations if name in configuration)),
        }
        for name in field_names
    ]
    configuration_scope = tuning.get("configuration_scope")
    if not isinstance(configuration_scope, dict):
        raise DecisionValidationError("final-tuning-domain", "final tuning requires a configuration_scope object", section.line)
    try:
        domain = validate_configuration_domain(profile, fields, configuration_scope)
    except ProfileValidationError as error:
        raise DecisionValidationError("final-tuning-profile-domain", f"profile legality rejects the tuning domain: {error.message}", section.line) from error
    domain_keys = {json.dumps(entry, sort_keys=True) for entry in domain}
    for configuration in configurations:
        if json.dumps(configuration, sort_keys=True) not in domain_keys:
            raise DecisionValidationError(
                "final-tuning-profile-domain",
                "a declared configuration is not covered by reviewed exact-scope profile legality",
                section.line,
            )

    for field in ("max_trials", "max_wall_seconds", "warmup", "repeat", "comparison_metric", "tie_rule"):
        if field not in tuning:
            raise DecisionValidationError("final-tuning-contract", f"final tuning requires {field}", section.line)
    if isinstance(tuning["max_trials"], bool) or not isinstance(tuning["max_trials"], int) or tuning["max_trials"] < 1:
        raise DecisionValidationError("final-tuning-contract", "max_trials must be a positive integer", section.line)
    if isinstance(tuning["max_wall_seconds"], bool) or not isinstance(tuning["max_wall_seconds"], (int, float)) or tuning["max_wall_seconds"] <= 0:
        raise DecisionValidationError("final-tuning-contract", "max_wall_seconds must be positive", section.line)
    for field in ("warmup", "repeat"):
        if isinstance(tuning[field], bool) or not isinstance(tuning[field], int) or tuning[field] < 0:
            raise DecisionValidationError("final-tuning-contract", f"{field} must be a non-negative integer", section.line)
    if tuning.get("pin_selected_config") is not True:
        raise DecisionValidationError("final-tuning-contract", "pin_selected_config must be true", section.line)
    return {
        "artifact_index": metadata.get("artifact_index"),
        "submission_snapshot_id": submission_snapshot_id,
        "anchors": anchors,
        "configurations": configurations,
        "fallback_configuration": fallback_configuration,
        "configuration_scope": configuration_scope,
        "max_trials": tuning["max_trials"],
        "max_wall_seconds": tuning["max_wall_seconds"],
        "warmup": tuning["warmup"],
        "repeat": tuning["repeat"],
        "mutation_reset": tuning.get("mutation_reset"),
        "comparison_metric": tuning["comparison_metric"],
        "tie_rule": tuning["tie_rule"],
        "pin_selected_config": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", type=Path, help="decision Markdown artifact")
    parser.add_argument("--expected-profile", help="required Metadata target_profile (schema-v1)")
    parser.add_argument("--expected-implementation-profile", help="required snapshot implementation profile (schema-v2)")
    parser.add_argument("--project-root", type=Path, help="project root for schema-v2 artifact resolution")
    args = parser.parse_args(argv)

    try:
        normalized = validate_decision(
            args.decision,
            args.expected_profile,
            project_root=args.project_root,
            expected_implementation_profile=args.expected_implementation_profile,
        )
    except DecisionValidationError as error:
        print(
            f"{args.decision}:{error.line}: {error.code}: {error.message}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(normalized, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

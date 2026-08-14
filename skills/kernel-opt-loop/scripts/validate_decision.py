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


def validate_decision(path: Path, expected_profile: str | None = None) -> dict[str, object]:
    """Validate *path* and return its normalized machine-readable contract."""

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
    unknown_sections = [heading for heading in sections if heading not in REQUIRED_SECTIONS]
    if unknown_sections:
        first = sections[unknown_sections[0]]
        raise DecisionValidationError(
            "section-unknown",
            f"unknown H2 section {first.heading!r}",
            first.line,
        )

    metadata = parse_single_json_block(sections["Metadata"])
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", type=Path, help="decision Markdown artifact")
    parser.add_argument("--expected-profile", help="required Metadata target_profile")
    args = parser.parse_args(argv)

    try:
        normalized = validate_decision(args.decision, args.expected_profile)
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

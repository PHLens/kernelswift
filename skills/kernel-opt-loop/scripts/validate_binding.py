"""Deterministic Sketch-to-source binding conformance.

The binding ledger proves source-level conformance to the Decision. Observed
lowering is a separate Verifier-owned claim. The first production source
analyzer is ``python-ast-triton``; a C-like profile may declare a future
analyzer that is explicitly unavailable rather than misclassified.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Callable

from vnext_common import (
    ContractValidationError,
    load_json_document,
    require_relative_artifact,
    sha256_file,
    validate_source_span,
)
from validate_profile import ProfileValidationError, require_capability


class BindingValidationError(ContractValidationError):
    pass


RELATIONS = frozenset({"implemented-by", "fused-into", "expanded-into", "elided-by"})
COVERED_KINDS = frozenset({"load", "compute", "store"})


def _error(code: str, message: str, path: Path | None = None) -> BindingValidationError:
    return BindingValidationError(code, message, path)


def validate_binding(
    binding_path: Path,
    *,
    project_root: Path,
    sketch_result: Mapping[str, Any],
    profile: Mapping[str, Any],
    candidate_path: Path,
    accepted_candidate_path: Path | None = None,
    final_tuning_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the binding ledger against the candidate, Sketch, and profile."""
    project_root = Path(project_root)
    binding = load_json_document(Path(binding_path), artifact="binding")
    candidate_path = Path(candidate_path).resolve()
    _validate_hashes(binding, project_root, candidate_path, sketch_result)
    _validate_final_tuning(binding, candidate_path, accepted_candidate_path, final_tuning_contract, profile)
    analyzer_name, binding_model, analyzer = _select_source_analyzer(profile)
    source = candidate_path.read_text(encoding="utf-8")
    source_symbols = analyzer(source)
    normalized = _validate_bindings(binding["bindings"], sketch_result, source_symbols, source, profile, binding["candidate_path"])
    _validate_required_coverage(normalized, sketch_result["statement_index"])
    required_hint_bindings = _validate_required_hints(binding, sketch_result, profile)
    return {
        "valid": True,
        "source_analyzer": analyzer_name,
        "binding_model": binding_model,
        "coverage": sorted(_coverage(normalized)),
        "source_symbols": source_symbols,
        "required_hint_bindings": required_hint_bindings,
        "bindings": normalized,
    }


def _validate_hashes(
    binding: Mapping[str, Any],
    project_root: Path,
    candidate_path: Path,
    sketch_result: Mapping[str, Any],
) -> None:
    if binding.get("schema_version") != 1:
        raise _error("binding-schema-version", "binding schema_version must be 1")
    recorded_path = binding.get("candidate_path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise _error("binding-candidate-path", "binding requires a candidate_path")
    candidate_file = require_relative_artifact(project_root, recorded_path)
    if candidate_file.resolve() != candidate_path:
        raise _error("binding-candidate-path", "binding candidate_path does not match the supplied candidate")
    recorded_sha = binding.get("candidate_sha256")
    if not isinstance(recorded_sha, str) or sha256_file(candidate_path) != recorded_sha:
        raise _error("binding-candidate-stale", "binding candidate_sha256 does not match the candidate file")
    sketch_sha = binding.get("sketch_sha256")
    if not isinstance(sketch_sha, str):
        raise _error("binding-sketch-hash", "binding requires sketch_sha256")
    decision_sha = binding.get("decision_sha256")
    if not isinstance(decision_sha, str):
        raise _error("binding-decision-hash", "binding requires decision_sha256")


def _validate_final_tuning(
    binding: Mapping[str, Any],
    candidate_path: Path,
    accepted_candidate_path: Path | None,
    final_tuning_contract: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
) -> None:
    if final_tuning_contract is None:
        return
    if accepted_candidate_path is None:
        raise _error("binding-final-tuning-accepted", "final tuning requires the accepted candidate path")
    if binding.get("artifact_kind") != "submission-finalization":
        raise _error("binding-final-tuning-kind", "final tuning binding requires artifact_kind submission-finalization")
    expected_index = final_tuning_contract.get("artifact_index")
    if binding.get("artifact_index") != expected_index:
        raise _error("binding-final-tuning-index", "final tuning binding artifact_index must match the Decision")
    if "round" in binding:
        raise _error("binding-final-tuning-round", "final tuning binding must not carry a campaign round")
    _validate_config_only_diff(accepted_candidate_path, candidate_path, final_tuning_contract, profile)


def _validate_config_only_diff(
    accepted_candidate_path: Path,
    pinned_candidate_path: Path,
    final_tuning_contract: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    analyzer_name, _binding_model, analyzer = _select_source_analyzer(profile)
    accepted_source = accepted_candidate_path.read_text(encoding="utf-8")
    pinned_source = pinned_candidate_path.read_text(encoding="utf-8")
    tunable_names = {
        name
        for configuration in final_tuning_contract.get("configurations") or []
        for name in configuration
    }
    accepted_normalized = _normalize_config_source(accepted_source, tunable_names)
    pinned_normalized = _normalize_config_source(pinned_source, tunable_names)
    if accepted_normalized != pinned_normalized:
        raise _error("binding-final-tuning-semantic", "pinned candidate changes more than Decision-authorized configuration values")
    accepted_values = _config_values(accepted_source, tunable_names)
    pinned_values = _config_values(pinned_source, tunable_names)
    if accepted_values == pinned_values:
        return
    declared = {json.dumps(configuration, sort_keys=True) for configuration in final_tuning_contract.get("configurations") or []}
    for name in accepted_values:
        if accepted_values[name] != pinned_values[name]:
            changed = dict(accepted_values)
            changed[name] = pinned_values[name]
            if json.dumps(changed, sort_keys=True) not in declared:
                raise _error("binding-final-tuning-semantic", f"changed configuration {name!r} is not a declared candidate configuration")


def _config_values(source: str, tunable_names: set[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return values
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in tunable_names:
                    values[keyword.arg] = _literal_value(keyword.value)
            if node.args and isinstance(node.args[0], ast.Subscript) and isinstance(node.args[0].value, ast.Name):
                for keyword in node.keywords:
                    if keyword.arg in tunable_names:
                        values[keyword.arg] = _literal_value(keyword.value)
    return values


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return ast.unparse(node)


def _normalize_config_source(source: str, tunable_names: set[str]) -> str:
    tree = ast.parse(source)

    class ConfigNormalizer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            node = self.generic_visit(node)
            for keyword in node.keywords:
                if keyword.arg in tunable_names:
                    keyword.value = ast.Constant(value="__CONFIG_PLACEHOLDER__")
            return node

    tree = ConfigNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True)


def _select_source_analyzer(profile: Mapping[str, Any]) -> tuple[str, str, Callable[[str], dict[str, list[list[int]]]]]:
    source_conformance = profile.get("source_conformance") or {}
    analyzer_name = source_conformance.get("analyzer")
    binding_model = source_conformance.get("binding_model")
    if not isinstance(analyzer_name, str) or not isinstance(binding_model, str):
        raise _error("profile-source-conformance", "profile requires source_conformance analyzer and binding_model")
    if analyzer_name == "python-ast-triton":
        return analyzer_name, binding_model, _analyze_python_ast
    raise _error(
        "profile-source-analyzer-unavailable",
        f"source analyzer {analyzer_name!r} is declared but not implemented",
    )


def _analyze_python_ast(source: str) -> dict[str, list[list[int]]]:
    """Return dotted call symbol -> list of one-based [line, col, end_line, end_col] spans."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise _error("binding-source-syntax", f"candidate source is not valid Python: {error.msg}") from error
    symbols: dict[str, list[list[int]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        if dotted is None:
            continue
        span = [node.lineno, node.col_offset + 1, node.end_lineno, node.end_col_offset + 1]
        symbols.setdefault(dotted, []).append(span)
    return symbols


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return None
    return ".".join(reversed(parts))


def _validate_bindings(
    bindings: Any,
    sketch_result: Mapping[str, Any],
    source_symbols: Mapping[str, list[list[int]]],
    source: str,
    profile: Mapping[str, Any],
    candidate_relative_path: str,
) -> list[dict[str, Any]]:
    if not isinstance(bindings, list) or not bindings:
        raise _error("binding-list-empty", "binding ledger requires at least one binding")
    statement_index = sketch_result["statement_index"]
    profile_symbols = {
        (entry["contract_name"], entry["implementation_symbol"])
        for entry in profile["capability_matrix"]
    }
    normalized: list[dict[str, Any]] = []
    statement_binding_counts: dict[str, int] = {}
    span_use_counts: dict[tuple[int, int, int, int], int] = {}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise _error("binding-entry-invalid", "each binding must be an object")
        statement_id = binding.get("statement_id")
        if statement_id not in statement_index:
            raise _error("binding-statement-unknown", f"binding references unknown statement {statement_id!r}")
        relation = binding.get("relation")
        if relation not in RELATIONS:
            raise _error("binding-relation-invalid", f"binding relation {relation!r} is invalid")
        contract_name = binding.get("contract_name")
        implementation_symbol = binding.get("implementation_symbol")
        if not isinstance(contract_name, str) or not contract_name.strip():
            raise _error("binding-contract-invalid", f"binding {statement_id!r} requires a contract_name")
        if not isinstance(implementation_symbol, str) or not implementation_symbol.strip():
            raise _error("binding-symbol-invalid", f"binding {statement_id!r} requires an implementation_symbol")
        if (contract_name, implementation_symbol) not in profile_symbols:
            raise _error("binding-profile-mapping", f"profile does not map {contract_name!r} to {implementation_symbol!r}")
        spans = binding.get("source_spans")
        if not isinstance(spans, list) or not spans:
            raise _error("binding-spans-required", f"binding {statement_id!r} requires source_spans")
        reason = binding.get("reason")
        if relation == "elided-by":
            if not isinstance(reason, str) or not reason.strip():
                raise _error("binding-elision-reason", "elided-by bindings require a nonempty reason")
            replacement = binding.get("replacement_statement")
            if replacement not in statement_index:
                raise _error("binding-elision-replacement", "elided-by bindings require an existing replacement statement")
        elif len(spans) > 1 and (not isinstance(reason, str) or not reason.strip()):
            raise _error("binding-many-reason", f"binding {statement_id!r} requires a reason for multiple source spans")
        normalized_spans: list[dict[str, Any]] = []
        for span in spans:
            if not isinstance(span, dict):
                raise _error("binding-span-invalid", f"binding {statement_id!r} span must be an object")
            span_path = span.get("path")
            if span_path != candidate_relative_path:
                raise _error("binding-span-ownership", f"binding {statement_id!r} span path is outside candidate ownership")
            if relation != "elided-by":
                matches = source_symbols.get(implementation_symbol) or []
                span_key = (span["start"][0], span["start"][1], span["end"][0], span["end"][1])
                if span_key not in {tuple(match) for match in matches}:
                    raise _error("binding-source-primitive", f"declared symbol {implementation_symbol!r} is not called at the exact span")
                validate_source_span(source, span)
                span_use_counts[span_key] = span_use_counts.get(span_key, 0) + 1
            normalized_spans.append(span)
        statement_binding_counts[statement_id] = statement_binding_counts.get(statement_id, 0) + 1
        if statement_binding_counts[statement_id] > 1 and (not isinstance(reason, str) or not reason.strip()):
            raise _error("binding-many-reason", f"statement {statement_id!r} has multiple bindings and requires a reason")
        normalized.append(
            {
                "statement_id": statement_id,
                "relation": relation,
                "contract_name": contract_name,
                "implementation_symbol": implementation_symbol,
                "source_spans": normalized_spans,
                "status": binding.get("status", "implemented"),
                "reason": reason,
                "notes": binding.get("notes"),
                "evidence": binding.get("evidence") or [],
            }
        )
    for count in span_use_counts.values():
        if count > 1:
            shared = [b for b in bindings if any(1 for span in b.get("source_spans", []) if span_use_counts.get((span["start"][0], span["start"][1], span["end"][0], span["end"][1])) and count > 1)]
            if any(not isinstance(b.get("reason"), str) or not b["reason"].strip() for b in shared):
                raise _error("binding-many-reason", "a source span shared by multiple statements requires a reason")
    return normalized


def _validate_required_coverage(normalized: list[dict[str, Any]], statement_index: Mapping[str, Any]) -> None:
    covered = {binding["statement_id"] for binding in normalized}
    for statement_id, statement in statement_index.items():
        if statement["kind"] in COVERED_KINDS and statement_id not in covered:
            raise _error("binding-coverage-missing", f"required statement {statement_id!r} has no binding")


def _validate_required_hints(
    binding: Mapping[str, Any],
    sketch_result: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = binding.get("required_hint_bindings") or []
    if not isinstance(records, list):
        raise _error("binding-hints-invalid", "required_hint_bindings must be a list")
    required = set(sketch_result.get("required_hints") or [])
    recorded_names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise _error("binding-hints-invalid", "each required hint binding must be an object")
        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _error("binding-hints-invalid", "each required hint binding requires a name")
        recorded_names.add(name)
        contract_name = record.get("contract_name")
        signature = record.get("signature")
        if not isinstance(contract_name, str) or not contract_name.strip() or not isinstance(signature, dict):
            raise _error("binding-hints-invalid", f"required hint {name!r} requires contract_name and signature")
        try:
            require_capability(profile, contract_name, signature, "required")
        except ProfileValidationError as error:
            raise _error("binding-hint-capability", f"required hint {name!r} is not profile-supported: {error.message}") from error
        normalized.append({"name": name, "contract_name": contract_name, "signature": signature, "status": record.get("status", "implemented")})
    missing = required - recorded_names
    if missing:
        raise _error("binding-hint-missing", f"required hints have no binding record: {sorted(missing)}")
    return normalized


def _coverage(normalized: list[dict[str, Any]]) -> set[str]:
    return {binding["statement_id"] for binding in normalized}

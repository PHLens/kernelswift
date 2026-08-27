"""Structural and semantic validation for the typed Unified Sketch.

The Sketch is the normative JSON contract for the complete computation boundary
affected by one round. This checker is pure: it never imports Triton, never
executes kernels, and only reasons about normalized mappings.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from pathlib import Path
from typing import Any

from vnext_common import ContractValidationError, load_json_document


class SketchValidationError(ContractValidationError):
    pass


DECLARATION_KINDS = frozenset({"tensor", "tile", "scalar"})
OPERATION_KINDS = frozenset({"alloc", "load", "compute", "store"})
HINT_MODALITIES = frozenset({"required", "preferred", "exploratory"})
LOAD_OUTPUT_DECLARATIONS = frozenset({"tile", "scalar"})
STORE_OUTPUT_DECLARATIONS = frozenset({"tensor", "scalar"})


def _error(code: str, message: str, path: Path | None = None) -> SketchValidationError:
    return SketchValidationError(code, message, path)


def validate_sketch(path: Path, *, expected_round: str | None = None) -> dict[str, Any]:
    sketch = load_json_document(path, artifact="sketch")
    _validate_header(sketch, expected_round)
    declarations = _index_declarations(sketch["declarations"])
    statement_index = _index_statements(sketch["operations"], sketch["control"])
    value_definitions = _validate_ssa_and_operation_signatures(declarations, sketch["operations"])
    _validate_control_and_bounds(sketch["operations"], sketch["control"])
    effect_outputs = _validate_effects_and_aliases(sketch["effects"], sketch["operations"], value_definitions)
    hint_groups = _validate_hint_modalities(sketch["hints"])
    causal_node_ids = _validate_causal_nodes(sketch["causal_nodes"], sketch["operations"], effect_outputs)
    return {
        "valid": True,
        "sketch": sketch,
        "statement_index": statement_index,
        "value_definitions": value_definitions,
        "effect_outputs": effect_outputs,
        "causal_node_ids": causal_node_ids,
        **hint_groups,
    }


def _validate_header(sketch: Mapping[str, Any], expected_round: str | None) -> None:
    if sketch.get("schema_version") != 1:
        raise _error("sketch-schema-version", "sketch schema_version must be 1")
    if not isinstance(sketch.get("sketch_id"), str) or not sketch["sketch_id"].strip():
        raise _error("sketch-id-required", "sketch requires a nonempty sketch_id")
    round_value = sketch.get("round")
    if not isinstance(round_value, str) or not re.fullmatch(r"[0-9]{3}", round_value):
        raise _error("sketch-round-invalid", "sketch round must contain exactly three decimal digits")
    if expected_round is not None and round_value != expected_round:
        raise _error("sketch-round-mismatch", f"sketch round {round_value!r} does not match expected round {expected_round!r}")


def _index_declarations(declarations: list[Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(declarations, list) or not declarations:
        raise _error("sketch-declarations-required", "sketch requires at least one declaration")
    indexed: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise _error("sketch-declaration-invalid", "each declaration must be an object")
        identifier = declaration.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise _error("sketch-declaration-id-required", "each declaration requires a nonempty id")
        if identifier in indexed:
            raise _error("sketch-duplicate-declaration", f"duplicate declaration id {identifier!r}")
        if declaration.get("kind") not in DECLARATION_KINDS:
            raise _error("sketch-declaration-kind-invalid", f"declaration {identifier!r} kind must be tensor|tile|scalar")
        for field in ("shape", "dtype", "layout", "memory"):
            value = declaration.get(field)
            if field == "shape":
                if not isinstance(value, list) or not value:
                    raise _error("sketch-declaration-shape-invalid", f"declaration {identifier!r} requires a nonempty shape")
            elif not isinstance(value, str) or not value.strip():
                raise _error("sketch-declaration-field-invalid", f"declaration {identifier!r} requires nonempty {field}")
        indexed[identifier] = declaration
    return indexed


def _index_statements(operations: list[Any], control: list[Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(operations, list) or not operations:
        raise _error("sketch-operations-required", "sketch requires at least one operation")
    if not isinstance(control, list):
        raise _error("sketch-control-invalid", "sketch control must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for operation in operations:
        if not isinstance(operation, dict):
            raise _error("sketch-operation-invalid", "each operation must be an object")
        identifier = operation.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise _error("sketch-statement-id-required", "each operation requires a nonempty id")
        if identifier in indexed:
            raise _error("sketch-duplicate-statement", f"duplicate statement id {identifier!r}")
        if operation.get("kind") not in OPERATION_KINDS:
            raise _error("sketch-operation-kind-invalid", f"operation {identifier!r} kind must be alloc|load|compute|store")
        for field in ("inputs", "outputs"):
            value = operation.get(field)
            if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
                raise _error("sketch-operation-values-invalid", f"operation {identifier!r} requires nonempty {field}")
        effects = operation.get("effects")
        if not isinstance(effects, dict) or set(effects) != {"reads", "writes"}:
            raise _error("sketch-operation-effect-invalid", f"operation {identifier!r} requires exact effects.reads/effects.writes lists; missing operation effect declaration")
        for field in ("reads", "writes"):
            if not isinstance(effects.get(field), list) or any(not isinstance(item, str) for item in effects[field]):
                raise _error("sketch-operation-effect-invalid", f"operation {identifier!r} effects.{field} must be a string list")
        indexed[identifier] = operation
    return indexed


def _validate_ssa_and_operation_signatures(
    declarations: Mapping[str, dict[str, Any]],
    operations: list[Any],
) -> dict[str, str]:
    """Validate def-use chains and return value -> defining statement id."""
    value_definitions: dict[str, str] = {identifier: identifier for identifier in declarations}
    operation_outputs: set[str] = set()

    for operation in operations:
        identifier = operation["id"]
        kind = operation["kind"]
        for value in operation["inputs"]:
            if value not in value_definitions:
                raise _error("sketch-undefined-value", f"operation {identifier!r} consumes undefined value {value!r}")
        for value in operation["outputs"]:
            if value in operation_outputs:
                raise _error("sketch-duplicate-definition", f"duplicate value definition {value!r} across operations")
            if value in declarations:
                _validate_output_declaration(identifier, kind, value, declarations[value])
                continue
            operation_outputs.add(value)
            value_definitions[value] = identifier
        _validate_operation_signatures(identifier, kind, operation, declarations, value_definitions)

    return value_definitions


def _validate_output_declaration(
    statement_id: str,
    kind: str,
    value: str,
    declaration: Mapping[str, Any],
) -> None:
    if kind == "load" and declaration["kind"] not in LOAD_OUTPUT_DECLARATIONS:
        raise _error("sketch-output-declaration-invalid", f"load {statement_id!r} may fill only a tile or scalar declaration, not {value!r}")
    if kind == "store" and declaration["kind"] not in STORE_OUTPUT_DECLARATIONS:
        raise _error("sketch-output-declaration-invalid", f"store {statement_id!r} may write only a tensor or scalar declaration, not {value!r}")
    if kind == "compute":
        raise _error("sketch-redefinition", f"compute {statement_id!r} may not redefine declaration {value!r}")


def _validate_operation_signatures(
    statement_id: str,
    kind: str,
    operation: Mapping[str, Any],
    declarations: Mapping[str, dict[str, Any]],
    value_definitions: Mapping[str, str],
) -> None:
    if kind != "compute":
        return
    resolved: list[dict[str, Any]] = []
    for value in (*operation["inputs"], *operation["outputs"]):
        declaration = declarations.get(value)
        if declaration is not None:
            resolved.append(declaration)
    if not resolved:
        return
    if "conversion" in operation:
        return
    baseline = _declaration_shape(resolved[0])
    for declaration in resolved[1:]:
        if _declaration_shape(declaration) != baseline:
            raise _error(
                "sketch-edge-type-mismatch",
                f"compute {statement_id!r} edges must agree on dtype, layout, and memory unless conversion is declared",
            )


def _declaration_shape(declaration: Mapping[str, Any]) -> tuple[str, str, str]:
    return (declaration["dtype"], declaration["layout"], declaration["memory"])


def _validate_control_and_bounds(operations: list[Any], control: list[Any]) -> None:
    guard_conditions = [
        entry.get("condition")
        for entry in control
        if isinstance(entry, dict) and entry.get("kind") == "guard" and isinstance(entry.get("condition"), str)
    ]
    parallel_variables = [
        entry.get("variable")
        for entry in control
        if isinstance(entry, dict) and entry.get("kind") == "parallel" and isinstance(entry.get("variable"), str)
    ]

    def mask_is_connected(mask: str) -> bool:
        if mask in guard_conditions:
            return True
        return any(re.search(rf"\b{re.escape(variable)}\b", mask) for variable in parallel_variables)

    for operation in operations:
        kind = operation["kind"]
        if kind not in {"load", "store"}:
            continue
        index_domain = operation.get("index_domain")
        mask = operation.get("mask")
        if not isinstance(index_domain, str) or not index_domain.strip():
            raise _error("sketch-index-unbounded", f"{kind} {operation['id']!r} requires a nonempty bounded index_domain")
        if not isinstance(mask, str) or not mask.strip() or not mask_is_connected(mask):
            raise _error("sketch-index-unbounded", f"{kind} {operation['id']!r} requires a mask connected to a guard or parallel domain")


def _validate_effects_and_aliases(
    effects: Any,
    operations: list[Any],
    value_definitions: Mapping[str, str],
) -> list[str]:
    if not isinstance(effects, dict):
        raise _error("sketch-effects-invalid", "sketch effects must be an object")
    for field in ("outputs", "mutations", "aliases"):
        if not isinstance(effects.get(field), list) or any(not isinstance(item, str) for item in effects[field]):
            raise _error("sketch-effects-invalid", f"sketch effects.{field} must be a string list")
    declared = set(effects["outputs"]) | set(effects["mutations"]) | set(effects["aliases"])

    for operation in operations:
        if operation["kind"] != "store":
            continue
        for target in operation["outputs"]:
            if target not in declared:
                raise _error("sketch-effect-undeclared", f"store {operation['id']!r} target {target!r} must be declared in effects.outputs, effects.mutations, or effects.aliases; effect undeclared")

    for alias in effects["aliases"]:
        if not isinstance(alias, (list, dict)):
            raise _error("sketch-alias-invalid", "each effect alias must name a source and a target")
        if isinstance(alias, list):
            names = alias
            if len(names) != 2 or any(not isinstance(name, str) or not name for name in names):
                raise _error("sketch-alias-invalid", "each effect alias must name exactly a source and a target")
        else:
            names = [alias.get("source"), alias.get("target")]
            if any(not isinstance(name, str) or not name for name in names):
                raise _error("sketch-alias-invalid", "each effect alias must name a source and a target")
        for name in names:
            if name not in value_definitions:
                raise _error("sketch-alias-undefined", f"alias names undefined value {name!r}")

    return list(effects["outputs"])


def _validate_hint_modalities(hints: Any) -> dict[str, list[str]]:
    if not isinstance(hints, list):
        raise _error("sketch-hints-invalid", "sketch hints must be a list")
    groups: dict[str, list[str]] = {"required_hints": [], "preferred_hints": [], "exploratory_hints": []}
    seen: set[str] = set()
    for hint in hints:
        if not isinstance(hint, dict):
            raise _error("sketch-hint-invalid", "each hint must be an object")
        name = hint.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _error("sketch-hint-invalid", "each hint requires a nonempty name")
        if name in seen:
            raise _error("sketch-hint-duplicate", f"duplicate hint name {name!r}")
        seen.add(name)
        modality = hint.get("modality")
        if modality not in HINT_MODALITIES:
            raise _error("sketch-hint-modality-invalid", f"hint {name!r} has invalid hint modality; modality must be required|preferred|exploratory")
        groups[f"{modality}_hints"].append(name)
    return groups


def _validate_causal_nodes(
    causal_nodes: Any,
    operations: list[Any],
    effect_outputs: list[str],
) -> list[str]:
    if not isinstance(causal_nodes, list):
        raise _error("sketch-causal-invalid", "sketch causal_nodes must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for node in causal_nodes:
        if not isinstance(node, dict):
            raise _error("sketch-causal-invalid", "each causal node must be an object")
        identifier = node.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise _error("sketch-causal-invalid", "each causal node requires a nonempty id")
        if identifier in indexed:
            raise _error("sketch-causal-duplicate", f"duplicate causal node id {identifier!r}")
        if not isinstance(node.get("kind"), str) or not node["kind"].strip():
            raise _error("sketch-causal-invalid", f"causal node {identifier!r} requires a nonempty kind")
        if not isinstance(node.get("expected"), str) or not node["expected"].strip():
            raise _error("sketch-causal-invalid", f"causal node {identifier!r} requires a nonempty expected outcome")
        indexed[identifier] = node

    referenced: set[str] = set()
    for operation in operations:
        for node_id in operation.get("causal_nodes") or []:
            if node_id not in indexed:
                raise _error("sketch-causal-undefined", f"operation {operation['id']!r} references undefined causal node {node_id!r}")
            referenced.add(node_id)
    for node_id in effect_outputs:
        if node_id in indexed:
            referenced.add(node_id)

    for identifier in indexed:
        if identifier not in referenced:
            raise _error(
                "sketch-causal-unreferenced",
                f"causal node {identifier!r} is referenced by no operation or output observable",
            )
    return list(indexed)

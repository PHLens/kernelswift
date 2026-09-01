"""Lightweight result validation for the BI150 value probe."""

from __future__ import annotations

from collections.abc import Mapping

FINAL_CLASSIFICATIONS = {
    "valuable",
    "technically-valid-low-value",
    "perturbation-invalid",
    "unsupported",
    "inconclusive",
}
EXPERIMENT_STATUSES = {"valid", "invalid", "unsupported", "inconclusive"}
PREFLIGHT_STATUSES = {"valid", "invalid", "unsupported", "inconclusive"}
REGION_STATUSES = {"observed", "unavailable", "invalid"}
MEASUREMENT_SEMANTICS = {"execution-duration", "issue-window"}


class DocumentValidationError(ValueError):
    """Raised when a result document violates the small experiment contract."""


def _mapping(value: object, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise DocumentValidationError(f"{path} must be an object")
    return value


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise DocumentValidationError(f"{path} must be a non-empty string")
    return value


def _required(payload: Mapping, key: str, path: str = "$") -> object:
    if key not in payload:
        raise DocumentValidationError(f"{path}.{key} is required")
    return payload[key]


def validate_final_classification(value: str) -> None:
    if value not in FINAL_CLASSIFICATIONS:
        allowed = ", ".join(sorted(FINAL_CLASSIFICATIONS))
        raise DocumentValidationError(
            f"final_classification must be one of: {allowed}"
        )


def _validate_region(region: object, index: int) -> None:
    path = f"$.regions[{index}]"
    document = _mapping(region, path)
    status = _nonempty_string(_required(document, "status", path), f"{path}.status")
    if status not in REGION_STATUSES:
        raise DocumentValidationError(f"{path}.status is unsupported: {status}")
    cause = _nonempty_string(_required(document, "cause", path), f"{path}.cause")
    semantics = document.get("measurement_semantics")
    if semantics is not None and semantics not in MEASUREMENT_SEMANTICS:
        raise DocumentValidationError(
            f"{path}.measurement_semantics is unsupported: {semantics}"
        )
    numeric_cycle_fields = [
        key for key in document if key.startswith("raw_cycle_") or key == "estimated_us"
    ]
    if status in {"unavailable", "invalid"} and numeric_cycle_fields:
        raise DocumentValidationError(
            f"{path}: {status} region must not contain cycle fields: "
            + ", ".join(sorted(numeric_cycle_fields))
        )
    if status == "observed" and "raw_cycle_delta" not in document:
        raise DocumentValidationError(f"{path}.raw_cycle_delta is required")
    if status == "observed" and cause != "none":
        raise DocumentValidationError(f"{path}.cause must be 'none' when observed")


def _validate_experiment_result(payload: Mapping) -> None:
    _nonempty_string(_required(payload, "experiment_id"), "$.experiment_id")
    environment = _mapping(_required(payload, "environment"), "$.environment")
    _nonempty_string(_required(environment, "route_c_commit", "$.environment"), "$.environment.route_c_commit")
    _mapping(_required(payload, "variant"), "$.variant")
    _mapping(_required(payload, "source"), "$.source")
    instrumentation = _mapping(
        _required(payload, "instrumentation"), "$.instrumentation"
    )
    semantics = instrumentation.get("measurement_semantics")
    if semantics is not None and semantics not in MEASUREMENT_SEMANTICS:
        raise DocumentValidationError(
            "$.instrumentation.measurement_semantics is unsupported: "
            f"{semantics}"
        )
    status = _nonempty_string(
        _required(payload, "experiment_status"), "$.experiment_status"
    )
    if status not in EXPERIMENT_STATUSES:
        raise DocumentValidationError(f"$.experiment_status is unsupported: {status}")
    regions = payload.get("regions", [])
    if not isinstance(regions, list):
        raise DocumentValidationError("$.regions must be an array")
    for index, region in enumerate(regions):
        _validate_region(region, index)
    causes = payload.get("status_causes", [])
    if not isinstance(causes, list) or not all(isinstance(item, str) for item in causes):
        raise DocumentValidationError("$.status_causes must be an array of strings")


def _validate_preflight_result(payload: Mapping) -> None:
    environment = _mapping(_required(payload, "environment"), "$.environment")
    _nonempty_string(_required(environment, "route_c_commit", "$.environment"), "$.environment.route_c_commit")
    status = _nonempty_string(_required(payload, "status"), "$.status")
    if status not in PREFLIGHT_STATUSES:
        raise DocumentValidationError(f"$.status is unsupported: {status}")
    causes = _required(payload, "causes")
    if not isinstance(causes, list) or not all(isinstance(item, str) for item in causes):
        raise DocumentValidationError("$.causes must be an array of strings")


def validate_document(payload: dict) -> None:
    document = _mapping(payload, "$")
    document_type = _nonempty_string(
        _required(document, "document_type"), "$.document_type"
    )
    if document_type == "experiment-result":
        _validate_experiment_result(document)
    elif document_type == "preflight-result":
        _validate_preflight_result(document)
    else:
        raise DocumentValidationError(
            f"$.document_type is unsupported: {document_type}"
        )


def document_route_c_commit(payload: dict) -> str:
    document = _mapping(payload, "$")
    if "route_c_commit" in document:
        return _nonempty_string(document["route_c_commit"], "$.route_c_commit")
    environment = _mapping(_required(document, "environment"), "$.environment")
    return _nonempty_string(
        _required(environment, "route_c_commit", "$.environment"),
        "$.environment.route_c_commit",
    )

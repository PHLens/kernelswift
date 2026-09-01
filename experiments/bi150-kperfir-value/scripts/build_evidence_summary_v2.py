#!/usr/bin/env python3
"""Freeze the terminal external-clock Route C evidence as a compact v2 summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
LIB_ROOT = EXPERIMENT_ROOT / "lib"
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

from result_contract import document_route_c_commit, validate_document
from source_guard import ACCEPTED_SOURCE_HASHES

EVIDENCE_COMMIT = "8e4d99b89407ecb3d35bac1c276d3cc73de27699"
HELPER_RELATIVE_PATH = Path(
    "experiments/bi150-kperfir-value/artifacts/external-clock/helper/clock-helper.json"
)
BITCODE_RELATIVE_PATH = Path(
    "experiments/bi150-kperfir-value/artifacts/external-clock/helper/corex-clock.bc"
)
SYNTHETIC_RELATIVE_PATH = Path(
    "experiments/bi150-kperfir-value/artifacts/external-clock/synthetic/result.json"
)
ATTENTION_RELATIVE_PATH = Path(
    "experiments/bi150-kperfir-value/artifacts/external-clock/attention/result.json"
)
OUTPUT_RELATIVE_PATH = Path(
    "experiments/bi150-kperfir-value/evidence-summary-v2.json"
)
TERMINAL_CAUSE = "end-dependency-optimized-away"
REQUIRED_LIMITATIONS = {
    "program-level-only",
    "issue-window-not-execution-duration",
}
SYNTHETIC_ARTIFACT_PREFIX = Path("artifacts/external-clock/synthetic")


class EvidenceSummaryV2Error(ValueError):
    """Raised when raw external-clock evidence is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceSummaryV2Error(f"evidence document must be an object: {path}")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceSummaryV2Error(f"{label} must be an object")
    return value


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise EvidenceSummaryV2Error(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def _require_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceSummaryV2Error(f"{label} must be a lowercase SHA256")
    return value


def _require_approved_ledger(value: object, label: str) -> dict[str, str]:
    approved = dict(ACCEPTED_SOURCE_HASHES)
    if value != approved:
        raise EvidenceSummaryV2Error(f"{label} source hash ledger is not approved")
    return approved


def _resolve_static_artifact(synthetic_path: Path, recorded_path: str) -> Path:
    relative = Path(recorded_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceSummaryV2Error("linked LLIR artifact path must be repository-relative")
    try:
        suffix = relative.relative_to(SYNTHETIC_ARTIFACT_PREFIX)
    except ValueError:
        return Path(synthetic_path).parent / relative
    return Path(synthetic_path).parent / suffix


def _validate_helper(
    helper_path: Path, expected_commit: str
) -> tuple[dict[str, Any], Path, str]:
    helper_path = Path(helper_path)
    helper = load_json(helper_path)
    _require_equal(helper.get("document_type"), "corex-clock-helper", "helper type")
    _require_equal(helper.get("status"), "valid", "helper status")
    _require_equal(helper.get("target"), "ivcore11", "helper target")
    _require_equal(
        helper.get("target_triple"), "bi-iluvatar-ilurt", "helper target triple"
    )

    ir_checks = _mapping(helper.get("ir_checks"), "helper ir_checks")
    if not ir_checks or not all(value is True for value in ir_checks.values()):
        raise EvidenceSummaryV2Error("helper IR checks are not all valid")

    bitcode_name = helper.get("bitcode_path")
    if not isinstance(bitcode_name, str) or Path(bitcode_name).name != bitcode_name:
        raise EvidenceSummaryV2Error("helper bitcode_path must be a local filename")
    bitcode_path = helper_path.parent / bitcode_name
    if not bitcode_path.is_file():
        raise EvidenceSummaryV2Error(f"helper bitcode is missing: {bitcode_path}")
    expected_bitcode_hash = _require_sha(
        helper.get("bitcode_sha256"), "helper bitcode_sha256"
    )
    _require_equal(
        sha256_file(bitcode_path), expected_bitcode_hash, "helper bitcode SHA256"
    )
    _require_sha(helper.get("source_sha256"), "helper source_sha256")

    absolute_path = helper.get("bitcode_absolute_path")
    if not isinstance(absolute_path, str) or expected_commit not in absolute_path:
        raise EvidenceSummaryV2Error(
            "helper bitcode path does not bind the expected route commit"
        )
    commands = _mapping(helper.get("commands"), "helper commands")
    for command_name in ("bitcode", "text_ir"):
        command = commands.get(command_name)
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise EvidenceSummaryV2Error(
                f"helper commands.{command_name} must be an array of strings"
            )
        if not any(expected_commit in item for item in command):
            raise EvidenceSummaryV2Error(
                f"helper commands.{command_name} does not bind the route commit"
            )

    return helper, bitcode_path, sha256_file(helper_path)


def _validate_synthetic(
    synthetic_path: Path,
    helper: Mapping[str, Any],
    helper_metadata_hash: str,
    expected_commit: str,
) -> tuple[dict[str, Any], Path]:
    synthetic_path = Path(synthetic_path)
    synthetic = load_json(synthetic_path)
    validate_document(synthetic)
    _require_equal(
        document_route_c_commit(synthetic), expected_commit, "synthetic route commit"
    )
    _require_approved_ledger(
        _mapping(synthetic.get("source"), "synthetic source").get(
            "accepted_source_hashes"
        ),
        "synthetic",
    )

    source_helper = _mapping(
        _mapping(synthetic.get("source"), "synthetic source").get("clock_helper"),
        "synthetic clock helper",
    )
    _require_equal(
        source_helper.get("bitcode_sha256"),
        helper.get("bitcode_sha256"),
        "synthetic/helper bitcode SHA256",
    )
    _require_equal(
        source_helper.get("source_sha256"),
        helper.get("source_sha256"),
        "synthetic/helper source SHA256",
    )
    _require_equal(
        source_helper.get("metadata_sha256"),
        helper_metadata_hash,
        "synthetic/helper metadata SHA256",
    )
    _require_equal(source_helper.get("target"), helper.get("target"), "helper target")
    _require_equal(
        source_helper.get("target_triple"),
        helper.get("target_triple"),
        "helper target triple",
    )

    instrumentation = _mapping(
        synthetic.get("instrumentation"), "synthetic instrumentation"
    )
    _require_equal(
        instrumentation.get("measurement_semantics"),
        "issue-window",
        "synthetic measurement semantics",
    )
    selected_pids = instrumentation.get("selected_pids")
    if selected_pids != [0]:
        raise EvidenceSummaryV2Error(
            "synthetic instrumentation must remain one selected program"
        )
    limitations = synthetic.get("limitations")
    if not isinstance(limitations, list) or not REQUIRED_LIMITATIONS.issubset(
        set(limitations)
    ):
        raise EvidenceSummaryV2Error(
            "synthetic evidence lacks program-level issue-window limitations"
        )

    audit = _mapping(
        synthetic.get("static_dependency_audit"), "static dependency audit"
    )
    artifact = _mapping(audit.get("artifact"), "static dependency artifact")
    _require_equal(artifact.get("status"), "observed", "linked LLIR status")
    recorded_path = artifact.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise EvidenceSummaryV2Error("linked LLIR artifact path is missing")
    llir_path = _resolve_static_artifact(synthetic_path, recorded_path)
    if not llir_path.is_file():
        raise EvidenceSummaryV2Error(f"linked LLIR artifact is missing: {llir_path}")
    expected_llir_hash = _require_sha(artifact.get("sha256"), "linked LLIR SHA256")
    _require_equal(sha256_file(llir_path), expected_llir_hash, "linked LLIR SHA256")
    _require_equal(
        llir_path.stat().st_size,
        artifact.get("byte_count"),
        "linked LLIR byte count",
    )
    return synthetic, llir_path


def _iter_attention_rows(attention: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("whole_span", "key_tiles", "deep_regions"):
        value = attention.get(key, [])
        if not isinstance(value, list):
            raise EvidenceSummaryV2Error(f"attention {key} must be an array")
        for index, row in enumerate(value):
            rows.append(_mapping(row, f"attention {key}[{index}]"))
    return rows


def _validate_attention_if_present(
    attention_path: Path | None,
    synthetic: Mapping[str, Any],
    helper: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any] | None:
    qualification = synthetic.get("qualification_status")
    if attention_path is None or not Path(attention_path).is_file():
        if qualification == "valid":
            raise EvidenceSummaryV2Error(
                "attention evidence is required after valid synthetic qualification"
            )
        return None

    attention = load_json(Path(attention_path))
    _require_equal(
        document_route_c_commit(attention), expected_commit, "attention route commit"
    )
    attention_helper = _mapping(attention.get("helper"), "attention helper")
    _require_equal(
        attention_helper.get("bitcode_sha256"),
        helper.get("bitcode_sha256"),
        "attention/helper bitcode SHA256",
    )
    limitations = attention.get("limitations")
    if not isinstance(limitations, list) or "program-level-only" not in limitations:
        raise EvidenceSummaryV2Error("attention evidence lacks program-level-only")
    for row in _iter_attention_rows(attention):
        if row.get("status") == "observed" and row.get(
            "measurement_semantics"
        ) != "issue-window":
            raise EvidenceSummaryV2Error(
                "observed attention rows must use issue-window semantics"
            )
    return attention


def build_evidence_summary_v2(
    helper_path: Path,
    synthetic_path: Path,
    attention_path: Path | None = None,
    *,
    expected_commit: str = EVIDENCE_COMMIT,
) -> dict[str, Any]:
    helper, bitcode_path, helper_metadata_hash = _validate_helper(
        helper_path, expected_commit
    )
    synthetic, llir_path = _validate_synthetic(
        synthetic_path, helper, helper_metadata_hash, expected_commit
    )
    attention = _validate_attention_if_present(
        attention_path, synthetic, helper, expected_commit
    )

    _require_equal(
        synthetic.get("qualification_status"),
        "inconclusive",
        "synthetic qualification status",
    )
    _require_equal(
        synthetic.get("experiment_status"),
        "inconclusive",
        "synthetic experiment status",
    )
    causes = synthetic.get("status_causes")
    if not isinstance(causes, list) or TERMINAL_CAUSE not in causes:
        raise EvidenceSummaryV2Error(
            f"synthetic evidence lacks terminal cause {TERMINAL_CAUSE}"
        )
    if synthetic.get("regions") != []:
        raise EvidenceSummaryV2Error(
            "terminal synthetic qualification must not contain cycle regions"
        )

    audit = _mapping(
        synthetic.get("static_dependency_audit"), "static dependency audit"
    )
    _require_equal(audit.get("status"), "optimized-away", "dependency audit status")
    _require_equal(
        audit.get("dependency_verified"), False, "dependency audit verification"
    )
    checks = _mapping(audit.get("linked_llir_checks"), "linked LLIR checks")
    _require_equal(
        checks.get("chain_dependency_verified"),
        False,
        "linked LLIR chain dependency",
    )
    if int(checks.get("clock_intrinsic_calls", 0)) < 2:
        raise EvidenceSummaryV2Error(
            "linked LLIR does not prove external clock intrinsic linkage"
        )
    _require_equal(
        checks.get("no_helper_runtime_calls"),
        True,
        "linked LLIR helper runtime calls",
    )

    helper_relative = HELPER_RELATIVE_PATH.as_posix()
    bitcode_relative = BITCODE_RELATIVE_PATH.as_posix()
    synthetic_relative = SYNTHETIC_RELATIVE_PATH.as_posix()
    llir_record = _mapping(audit.get("artifact"), "static dependency artifact")

    return {
        "document_type": "bi150-route-c-evidence-summary-v2",
        "evidence_commit": expected_commit,
        "overall_classification": "inconclusive",
        "causes": [TERMINAL_CAUSE],
        "stop_statement": (
            "Tasks 3 and 4 were not run because Stage A could not preserve an "
            "honest completion-dependent end marker."
        ),
        "accepted_source_hashes": dict(ACCEPTED_SOURCE_HASHES),
        "raw_evidence": {
            "helper_metadata": {
                "relative_path": helper_relative,
                "sha256": helper_metadata_hash,
            },
            "helper_bitcode": {
                "relative_path": bitcode_relative,
                "sha256": sha256_file(bitcode_path),
            },
            "synthetic": {
                "relative_path": synthetic_relative,
                "sha256": sha256_file(Path(synthetic_path)),
            },
            "linked_llir": {
                "relative_path": llir_record["path"],
                "sha256": sha256_file(llir_path),
                "byte_count": llir_path.stat().st_size,
            },
            "attention": {
                "status": "observed" if attention is not None else "not-run",
                "relative_path": ATTENTION_RELATIVE_PATH.as_posix(),
                "reason": (
                    None
                    if attention is not None
                    else "synthetic-qualification-inconclusive"
                ),
                "sha256": (
                    sha256_file(Path(attention_path)) if attention is not None else None
                ),
            },
        },
        "environment": dict(_mapping(synthetic.get("environment"), "environment")),
        "external_clock_capability": {
            "linkage": "proven",
            "clock_source": "CoreX CUDA C clock64 via external LLVM bitcode",
            "helper_status": helper["status"],
            "target": helper["target"],
            "target_triple": helper["target_triple"],
            "bitcode_sha256": helper["bitcode_sha256"],
            "linked_clock_intrinsic_calls": checks["clock_intrinsic_calls"],
            "helper_runtime_calls": checks.get("helper_runtime_calls", 0),
        },
        "synthetic_qualification": {
            "status": synthetic["qualification_status"],
            "status_causes": list(causes),
            "error": synthetic.get("error"),
            "chain_iters": list(synthetic.get("chain_iters", [])),
            "instrumentation": dict(
                _mapping(synthetic.get("instrumentation"), "instrumentation")
            ),
            "limitations": list(synthetic.get("limitations", [])),
            "static_dependency_audit": {
                "status": audit["status"],
                "mode": audit.get("mode"),
                "chain_iters": audit.get("chain_iters"),
                "compiled_hash": audit.get("compiled_hash"),
                "dependency_verified": audit["dependency_verified"],
                "linked_llir_checks": dict(checks),
            },
        },
        "runtime_history": {
            "noinline_helper": {
                "result": "first-kernel-execution-hung",
                "host_wait": "torch.cuda.synchronize",
                "termination": "process-terminated-after-approximately-seven-minutes",
            },
            "post_hang_gpu_context": {
                "result": "container-visible-context-unusable",
                "evidence": "minimal CUDA allocation stalled",
                "reset": "container reset unavailable because owning host PIDs were outside its PID namespace",
            },
            "later_smoke_attribution": (
                "unavailable: the post-hang GPU context was already unusable, so "
                "the later stall is not attributed to inline assembly"
            ),
        },
        "attention": {
            "status": "observed" if attention is not None else "not-run",
            "reason": (
                None
                if attention is not None
                else "stage-a-end-dependency-optimized-away"
            ),
            "cycle_record_count": 0 if attention is None else None,
        },
        "incremental_value": {
            "new_measurement_capability": (
                "partial: CoreX external bitcode linkage and clock intrinsic "
                "materialization were proven"
            ),
            "new_kernel_fact": "none: no attention cycle records were obtained",
            "new_optimization_decision": "none",
        },
        "authority_boundary": (
            "Diagnostic-only evidence; accepted kernels, official timings, and "
            "campaign authority are unchanged."
        ),
    }


def write_evidence_summary_v2(
    helper_path: Path,
    synthetic_path: Path,
    attention_path: Path | None,
    output_path: Path,
    *,
    expected_commit: str = EVIDENCE_COMMIT,
) -> dict[str, Any]:
    summary = build_evidence_summary_v2(
        helper_path,
        synthetic_path,
        attention_path,
        expected_commit=expected_commit,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--helper", type=Path, default=REPO_ROOT / HELPER_RELATIVE_PATH
    )
    parser.add_argument(
        "--synthetic", type=Path, default=REPO_ROOT / SYNTHETIC_RELATIVE_PATH
    )
    parser.add_argument(
        "--attention", type=Path, default=REPO_ROOT / ATTENTION_RELATIVE_PATH
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    write_evidence_summary_v2(
        args.helper,
        args.synthetic,
        args.attention,
        args.output,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

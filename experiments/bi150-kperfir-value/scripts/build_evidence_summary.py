#!/usr/bin/env python3
"""Build the small, versioned evidence summary for the BI150 Route C run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
LIB_ROOT = EXPERIMENT_ROOT / "lib"
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

from result_contract import document_route_c_commit, validate_document
from source_guard import ACCEPTED_SOURCE_HASHES

EVIDENCE_COMMIT = "e61443606746959ea537a20190308d20af93234c"
PREFLIGHT_RELATIVE_PATH = Path(
    "experiments/bi150-kperfir-value/artifacts/evidence/preflight/stage0-result.json"
)
CLOCK64_RELATIVE_PATH = Path(
    "experiments/bi150-kperfir-value/artifacts/evidence/clock64/result.json"
)
OUTPUT_RELATIVE_PATH = Path(
    "experiments/bi150-kperfir-value/evidence-summary.json"
)
OVERALL_CAUSES = [
    "final-isa-unavailable",
    "clock64-inline-asm-syntax-rejected",
]


class EvidenceSummaryError(ValueError):
    """Raised when the two raw evidence documents cannot form one summary."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceSummaryError(f"evidence document must be an object: {path}")
    return payload


def _require_approved_ledger(ledger: object, label: str) -> dict[str, str]:
    approved = dict(ACCEPTED_SOURCE_HASHES)
    if ledger != approved:
        raise EvidenceSummaryError(f"{label} source hash ledger is not approved")
    return approved


def _disassembly_outcomes(preflight: dict) -> list[dict]:
    outcomes = []
    for attempt in preflight["disassembly_attempts"]:
        command = attempt["command"]
        outcomes.append(
            {
                "tool": Path(command[0]).name,
                "returncode": attempt["returncode"],
                "valid": attempt["valid"],
                "function_body_present": attempt["function_body_present"],
                "stdout_byte_count": attempt["stdout_byte_count"],
                "error": attempt["stderr_tail"].strip() or None,
            }
        )
    return outcomes


def build_evidence_summary(preflight_path: Path, clock64_path: Path) -> dict:
    preflight_path = Path(preflight_path)
    clock64_path = Path(clock64_path)
    preflight = load_json(preflight_path)
    clock64 = load_json(clock64_path)

    validate_document(preflight)
    validate_document(clock64)

    preflight_commit = document_route_c_commit(preflight)
    clock64_commit = document_route_c_commit(clock64)
    if preflight_commit != clock64_commit:
        raise EvidenceSummaryError(
            "raw evidence commit mismatch: "
            f"preflight={preflight_commit}, clock64={clock64_commit}"
        )
    if preflight_commit != EVIDENCE_COMMIT:
        raise EvidenceSummaryError(
            f"unexpected evidence commit: {preflight_commit}; expected {EVIDENCE_COMMIT}"
        )

    approved_sources = _require_approved_ledger(
        preflight.get("accepted_sources"), "preflight"
    )
    _require_approved_ledger(
        clock64.get("source", {}).get("accepted_source_hashes"), "clock64"
    )

    if "final-isa-unavailable" not in preflight.get("causes", []):
        raise EvidenceSummaryError("preflight evidence lacks final-isa-unavailable")
    if "clock64-inline-asm-syntax-rejected" not in clock64.get(
        "status_causes", []
    ):
        raise EvidenceSummaryError(
            "clock64 evidence lacks clock64-inline-asm-syntax-rejected"
        )

    environment = preflight["environment"]
    stock_kernel = preflight["stock_kernel"]
    instrumentation = clock64["instrumentation"]

    return {
        "document_type": "bi150-route-c-evidence-summary",
        "evidence_commit": EVIDENCE_COMMIT,
        "overall_classification": "inconclusive",
        "causes": list(OVERALL_CAUSES),
        "stop_statement": (
            "Real attention and graph probes were not run because no "
            "interpretable clock path existed."
        ),
        "accepted_source_hashes": approved_sources,
        "raw_evidence": {
            "preflight": {
                "relative_path": PREFLIGHT_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(preflight_path),
            },
            "clock64": {
                "relative_path": CLOCK64_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(clock64_path),
            },
        },
        "environment": {
            "device": environment["device"],
            "corex": environment["corex"],
            "torch": environment["torch"],
            "triton": environment["triton"],
            "target": environment["target"],
            "warp_size": environment["warp_size"],
        },
        "preflight": {
            "status": preflight["status"],
            "correctness": preflight["correctness"],
            "resources": stock_kernel["resources"],
            "cuda_event_timing": preflight["cuda_event_timing"],
            "disassembly_outcomes": _disassembly_outcomes(preflight),
        },
        "clock64": {
            "status": clock64["experiment_status"],
            "status_causes": clock64["status_causes"],
            "attempted_clock_syntaxes": instrumentation[
                "attempted_clock_syntaxes"
            ],
            "measurement_semantics": instrumentation["measurement_semantics"],
            "chain_iters": clock64["chain_iters"],
            "worker_returncode": clock64["worker_returncode"],
            "error": clock64["error"],
        },
        "not_run": {
            "real_attention_probe": {
                "ran": False,
                "reason": "no-interpretable-clock-path",
            },
            "graph_replay_probe": {
                "ran": False,
                "reason": "no-interpretable-clock-path",
            },
        },
    }


def write_evidence_summary(
    preflight_path: Path, clock64_path: Path, output_path: Path
) -> dict:
    summary = build_evidence_summary(preflight_path, clock64_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=REPO_ROOT / PREFLIGHT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--clock64",
        type=Path,
        default=REPO_ROOT / CLOCK64_RELATIVE_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / OUTPUT_RELATIVE_PATH,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_evidence_summary(args.preflight, args.clock64, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

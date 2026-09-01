import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "lib"))

from build_evidence_summary_v2 import (
    EVIDENCE_COMMIT,
    EvidenceSummaryV2Error,
    build_evidence_summary_v2,
    write_evidence_summary_v2,
)
from source_guard import ACCEPTED_SOURCE_HASHES


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EvidenceFixture:
    def __init__(self, root: Path):
        self.root = root
        self.helper_dir = root / "helper"
        self.synthetic_dir = root / "synthetic"
        self.llir_dir = self.synthetic_dir / "linked-llir"
        self.helper_dir.mkdir(parents=True)
        self.llir_dir.mkdir(parents=True)

        self.bitcode_path = self.helper_dir / "corex-clock.bc"
        self.bitcode_path.write_bytes(b"corex-clock-bitcode\x00")
        self.bitcode_sha = sha256_bytes(self.bitcode_path.read_bytes())

        self.helper_path = self.helper_dir / "clock-helper.json"
        self.helper = {
            "document_type": "corex-clock-helper",
            "status": "valid",
            "target": "ivcore11",
            "target_triple": "bi-iluvatar-ilurt",
            "corex_root": "/usr/local/corex-4.4.0",
            "bitcode_path": "corex-clock.bc",
            "bitcode_absolute_path": (
                f"/tmp/kernelswift-route-c-{EVIDENCE_COMMIT}/"
                "experiments/bi150-kperfir-value/artifacts/external-clock/"
                "helper/corex-clock.bc"
            ),
            "bitcode_sha256": self.bitcode_sha,
            "source_sha256": "a" * 64,
            "ir_path": "corex-clock.ll",
            "ir_checks": {
                "start_symbol": True,
                "dependent_end_symbol": True,
                "start_clock64_intrinsic": True,
                "end_clock64_intrinsic": True,
                "start_alwaysinline": True,
                "end_alwaysinline": True,
            },
            "commands": {
                "bitcode": [
                    "/usr/local/corex-4.4.0/bin/clang++",
                    f"/tmp/kernelswift-route-c-{EVIDENCE_COMMIT}/corex_clock.cu",
                ],
                "text_ir": [
                    "/usr/local/corex-4.4.0/bin/clang++",
                    f"/tmp/kernelswift-route-c-{EVIDENCE_COMMIT}/corex_clock.cu",
                ],
            },
        }
        self.write_helper()

        self.llir_path = self.llir_dir / "clock64-static-dependency-audit.ll"
        self.llir_path.write_text(
            "target triple = \"bi-iluvatar-ilurt\"\n"
            "%start = call i64 @llvm.nvvm.read.ptx.sreg.clock64()\n"
            "%end = call i64 @llvm.nvvm.read.ptx.sreg.clock64()\n",
            encoding="utf-8",
        )
        self.llir_sha = hashlib.sha256(self.llir_path.read_bytes()).hexdigest()

        self.synthetic_path = self.synthetic_dir / "result.json"
        self.synthetic = self.synthetic_payload()
        self.write_synthetic()

    @property
    def helper_sha(self) -> str:
        return hashlib.sha256(self.helper_path.read_bytes()).hexdigest()

    def write_helper(self) -> None:
        self.helper_path.write_text(
            json.dumps(self.helper, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def synthetic_payload(self) -> dict:
        return {
            "document_type": "experiment-result",
            "experiment_id": "bi150-external-clock-qualification",
            "environment": {
                "route_c_commit": EVIDENCE_COMMIT,
                "device": "Iluvatar BI-V150",
                "corex": "4.4.0",
                "torch": "2.7.1",
                "triton": "3.1.0",
                "target": "GPUTarget(backend='cuda', arch=71, warp_size=64)",
            },
            "variant": {
                "execution_mode": "eager",
                "kernel_variant": "synthetic-external-clock-one-program",
                "num_warps": 1,
            },
            "source": {
                "accepted_source_hashes": dict(ACCEPTED_SOURCE_HASHES),
                "diagnostic_sha256": "b" * 64,
                "clock_helper": {
                    "bitcode_sha256": self.bitcode_sha,
                    "metadata_sha256": self.helper_sha,
                    "source_sha256": "a" * 64,
                    "symbols": [
                        "corex_clock64_start",
                        "corex_clock64_after_u64",
                    ],
                    "target": "ivcore11",
                    "target_triple": "bi-iluvatar-ilurt",
                },
            },
            "instrumentation": {
                "mode": "corex-external-bitcode-inline-control",
                "region_id": "dependency-chain",
                "selected_pids": [0],
                "time_unit": "raw-cycle",
                "storage": "one-program-generation-start-end",
                "measurement_semantics": "issue-window",
                "profile_words": 3,
                "completion_dependency": "token-dependent-control-dependency",
                "clock_symbols": [
                    "corex_clock64_start",
                    "corex_clock64_after_u64",
                ],
            },
            "chain_iters": [0, 16, 256],
            "warmup_runs": 20,
            "regions": [],
            "qualification_status": "inconclusive",
            "experiment_status": "inconclusive",
            "status_causes": ["end-dependency-optimized-away"],
            "error": (
                "compile-only linked LLIR did not preserve a chain-derived "
                "completion dependency before the end clock"
            ),
            "limitations": [
                "program-level-only",
                "issue-window-not-execution-duration",
                "final-isa-unavailable",
                "noinline-helper-runtime-hang-observed",
                "post-noinline-gpu-context-unavailable",
            ],
            "static_dependency_audit": {
                "status": "optimized-away",
                "mode": "compile-only-prelaunch",
                "chain_iters": 16,
                "compiled_hash": "c" * 64,
                "dependency_verified": False,
                "artifact": {
                    "status": "observed",
                    "path": (
                        "artifacts/external-clock/synthetic/linked-llir/"
                        "clock64-static-dependency-audit.ll"
                    ),
                    "sha256": self.llir_sha,
                    "byte_count": self.llir_path.stat().st_size,
                },
                "linked_llir_checks": {
                    "chain_dependency_verified": False,
                    "chain_operation_count": 0,
                    "clock_intrinsic_calls": 2,
                    "conditional_branch_count": 1,
                    "dependent_end_branch": True,
                    "end_clock_calls": 0,
                    "helper_definitions_retained": True,
                    "helper_runtime_calls": 0,
                    "inline_asm_calls": 0,
                    "intrinsic_count_ok": True,
                    "linked": False,
                    "no_helper_runtime_calls": True,
                    "no_inline_asm": True,
                    "required_chain_operations": 4,
                    "runtime_seed_dependency": False,
                    "start_before_end": False,
                    "token_branch_after_start": False,
                },
            },
        }

    def write_synthetic(self) -> None:
        self.synthetic_path.write_text(
            json.dumps(self.synthetic, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class EvidenceSummaryV2Tests(unittest.TestCase):
    def test_builds_terminal_inconclusive_summary_with_bound_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            summary = build_evidence_summary_v2(
                fixture.helper_path, fixture.synthetic_path
            )

            self.assertEqual("inconclusive", summary["overall_classification"])
            self.assertEqual(
                ["end-dependency-optimized-away"], summary["causes"]
            )
            self.assertEqual(
                "proven", summary["external_clock_capability"]["linkage"]
            )
            self.assertEqual(
                "inconclusive", summary["synthetic_qualification"]["status"]
            )
            self.assertEqual("not-run", summary["attention"]["status"])
            self.assertEqual(0, summary["attention"]["cycle_record_count"])
            self.assertEqual(
                "none: no attention cycle records were obtained",
                summary["incremental_value"]["new_kernel_fact"],
            )
            self.assertEqual(
                fixture.helper_sha,
                summary["raw_evidence"]["helper_metadata"]["sha256"],
            )
            self.assertEqual(
                fixture.llir_sha,
                summary["raw_evidence"]["linked_llir"]["sha256"],
            )
            self.assertNotIn("unsupported", json.dumps(summary))
            self.assertNotIn("technically-valid-low-value", json.dumps(summary))

    def test_rejects_unapproved_source_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            fixture.synthetic["source"]["accepted_source_hashes"][
                "auto_bench.py"
            ] = "0" * 64
            fixture.write_synthetic()
            with self.assertRaisesRegex(EvidenceSummaryV2Error, "not approved"):
                build_evidence_summary_v2(
                    fixture.helper_path, fixture.synthetic_path
                )

    def test_rejects_route_commit_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            fixture.synthetic["environment"]["route_c_commit"] = "different"
            fixture.write_synthetic()
            with self.assertRaisesRegex(EvidenceSummaryV2Error, "route commit"):
                build_evidence_summary_v2(
                    fixture.helper_path, fixture.synthetic_path
                )

    def test_rejects_helper_route_commit_binding_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            fixture.helper["bitcode_absolute_path"] = (
                "/tmp/kernelswift-route-c-different/corex-clock.bc"
            )
            fixture.write_helper()
            with self.assertRaisesRegex(EvidenceSummaryV2Error, "route commit"):
                build_evidence_summary_v2(
                    fixture.helper_path, fixture.synthetic_path
                )

    def test_rejects_helper_bitcode_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            fixture.bitcode_path.write_bytes(b"modified")
            with self.assertRaisesRegex(EvidenceSummaryV2Error, "bitcode SHA256"):
                build_evidence_summary_v2(
                    fixture.helper_path, fixture.synthetic_path
                )

    def test_rejects_helper_metadata_binding_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            fixture.synthetic["source"]["clock_helper"]["metadata_sha256"] = (
                "0" * 64
            )
            fixture.write_synthetic()
            with self.assertRaisesRegex(EvidenceSummaryV2Error, "metadata SHA256"):
                build_evidence_summary_v2(
                    fixture.helper_path, fixture.synthetic_path
                )

    def test_rejects_linked_llir_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            fixture.llir_path.write_text("modified\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceSummaryV2Error, "linked LLIR SHA256"):
                build_evidence_summary_v2(
                    fixture.helper_path, fixture.synthetic_path
                )

    def test_requires_program_level_issue_window_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            fixture.synthetic["instrumentation"]["measurement_semantics"] = (
                "execution-duration"
            )
            fixture.synthetic["limitations"].remove("program-level-only")
            fixture.write_synthetic()
            with self.assertRaises(EvidenceSummaryV2Error):
                build_evidence_summary_v2(
                    fixture.helper_path, fixture.synthetic_path
                )

    def test_missing_attention_is_rejected_after_valid_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            fixture.synthetic["qualification_status"] = "valid"
            fixture.synthetic["experiment_status"] = "valid"
            fixture.synthetic["status_causes"] = []
            fixture.write_synthetic()
            with self.assertRaisesRegex(
                EvidenceSummaryV2Error, "attention evidence is required"
            ):
                build_evidence_summary_v2(
                    fixture.helper_path, fixture.synthetic_path
                )

    def test_write_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvidenceFixture(Path(directory))
            first = fixture.root / "summary-a.json"
            second = fixture.root / "summary-b.json"
            first_summary = write_evidence_summary_v2(
                fixture.helper_path, fixture.synthetic_path, None, first
            )
            second_summary = write_evidence_summary_v2(
                fixture.helper_path, fixture.synthetic_path, None, second
            )
            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()

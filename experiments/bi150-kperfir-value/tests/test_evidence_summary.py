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

from build_evidence_summary import (
    EVIDENCE_COMMIT,
    EvidenceSummaryError,
    build_evidence_summary,
    write_evidence_summary,
)
from source_guard import ACCEPTED_SOURCE_HASHES


def preflight_payload() -> dict:
    return {
        "document_type": "preflight-result",
        "environment": {
            "route_c_commit": EVIDENCE_COMMIT,
            "device": "Iluvatar BI-V150",
            "corex": "4.4.0",
            "torch": "2.7.1",
            "triton": "3.1.0",
            "target": "GPUTarget(backend='cuda', arch=71, warp_size=64)",
            "warp_size": 64,
        },
        "accepted_sources": dict(ACCEPTED_SOURCE_HASHES),
        "status": "inconclusive",
        "causes": ["final-isa-unavailable"],
        "correctness": {"status": "pass", "bitwise_equal": True},
        "stock_kernel": {
            "resources": {
                "n_regs": 13,
                "n_spills": 0,
                "n_threads_attribute": 4096,
                "shared_bytes": 0,
            }
        },
        "cuda_event_timing": {
            "sample_count": 50,
            "median_us": 5.6,
            "p10_us": 5.5,
            "p90_us": 5.8,
            "coefficient_of_variation": 0.03,
        },
        "disassembly_attempts": [
            {
                "command": ["/opt/llvm-objdump", "-d", "stock.cubin"],
                "returncode": 1,
                "valid": False,
                "function_body_present": False,
                "stdout_byte_count": 80,
                "stderr_tail": "no disassembler for target",
            }
        ],
    }


def clock64_payload() -> dict:
    return {
        "document_type": "experiment-result",
        "experiment_id": "bi150-clock64-functional",
        "environment": {"route_c_commit": EVIDENCE_COMMIT},
        "variant": {
            "execution_mode": "eager",
            "kernel_variant": "synthetic-clock64-one-pair",
            "num_warps": 2,
        },
        "source": {
            "accepted_source_hashes": dict(ACCEPTED_SOURCE_HASHES),
            "diagnostic_sha256": "a" * 64,
        },
        "instrumentation": {
            "attempted_clock_syntaxes": ["mov.u64 $0, %clock64;"],
            "measurement_semantics": "issue-window",
        },
        "regions": [],
        "experiment_status": "inconclusive",
        "status_causes": ["clock64-inline-asm-syntax-rejected"],
        "chain_iters": [16, 256],
        "worker_returncode": 1,
        "error": "unknown token in expression",
    }


class EvidenceSummaryTests(unittest.TestCase):
    def write_raw_documents(self, directory: Path, *, preflight=None, clock64=None):
        preflight_path = directory / "stage0-result.json"
        clock64_path = directory / "clock64-result.json"
        preflight_path.write_text(
            json.dumps(preflight or preflight_payload(), sort_keys=True),
            encoding="utf-8",
        )
        clock64_path.write_text(
            json.dumps(clock64 or clock64_payload(), sort_keys=True),
            encoding="utf-8",
        )
        return preflight_path, clock64_path

    def test_builds_small_inconclusive_summary_with_raw_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight_path, clock64_path = self.write_raw_documents(root)
            output_path = root / "evidence-summary.json"
            summary = write_evidence_summary(
                preflight_path, clock64_path, output_path
            )

            self.assertEqual("inconclusive", summary["overall_classification"])
            self.assertEqual(
                [
                    "final-isa-unavailable",
                    "clock64-inline-asm-syntax-rejected",
                ],
                summary["causes"],
            )
            self.assertEqual(13, summary["preflight"]["resources"]["n_regs"])
            self.assertEqual(
                "mov.u64 $0, %clock64;",
                summary["clock64"]["attempted_clock_syntaxes"][0],
            )
            self.assertFalse(summary["not_run"]["real_attention_probe"]["ran"])
            self.assertFalse(summary["not_run"]["graph_replay_probe"]["ran"])
            self.assertEqual(
                "Real attention and graph probes were not run because no "
                "interpretable clock path existed.",
                summary["stop_statement"],
            )
            self.assertEqual(
                hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
                summary["raw_evidence"]["preflight"]["sha256"],
            )
            self.assertEqual(summary, json.loads(output_path.read_text()))

    def test_rejects_commit_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            clock64 = clock64_payload()
            clock64["environment"]["route_c_commit"] = "different"
            paths = self.write_raw_documents(Path(directory), clock64=clock64)
            with self.assertRaisesRegex(EvidenceSummaryError, "commit mismatch"):
                build_evidence_summary(*paths)

    def test_rejects_unapproved_source_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            preflight = copy.deepcopy(preflight_payload())
            preflight["accepted_sources"]["auto_bench.py"] = "0" * 64
            paths = self.write_raw_documents(Path(directory), preflight=preflight)
            with self.assertRaisesRegex(EvidenceSummaryError, "not approved"):
                build_evidence_summary(*paths)


if __name__ == "__main__":
    unittest.main()

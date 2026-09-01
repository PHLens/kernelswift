import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from result_contract import (
    DocumentValidationError,
    document_route_c_commit,
    validate_document,
    validate_final_classification,
)


def valid_experiment():
    return {
        "document_type": "experiment-result",
        "experiment_id": "bi150-synthetic-clock-short",
        "environment": {"route_c_commit": "abc123", "device": "BI-V150"},
        "variant": {"kernel_variant": "clock-short", "execution_mode": "eager"},
        "source": {"diagnostic_sha256": "deadbeef"},
        "instrumentation": {"measurement_semantics": "issue-window"},
        "regions": [
            {
                "pid": 0,
                "local_warp": 0,
                "status": "observed",
                "cause": "none",
                "measurement_semantics": "issue-window",
                "raw_cycle_delta": 42,
            }
        ],
        "status_causes": [],
        "experiment_status": "valid",
    }


class ResultContractTests(unittest.TestCase):
    def test_accepts_final_classifications(self):
        for value in (
            "valuable",
            "technically-valid-low-value",
            "perturbation-invalid",
            "unsupported",
            "inconclusive",
        ):
            validate_final_classification(value)

    def test_rejects_unknown_final_classification(self):
        with self.assertRaisesRegex(DocumentValidationError, "must be one of"):
            validate_final_classification("maybe")

    def test_accepts_basic_experiment_result(self):
        validate_document(valid_experiment())

    def test_unavailable_region_rejects_fabricated_cycle_value(self):
        payload = valid_experiment()
        payload["regions"][0] = {
            "status": "unavailable",
            "cause": "generation-mismatch",
            "raw_cycle_delta": 0,
        }
        with self.assertRaisesRegex(
            DocumentValidationError, "must not contain cycle fields"
        ):
            validate_document(payload)

    def test_accepts_basic_preflight_result(self):
        payload = {
            "document_type": "preflight-result",
            "environment": {"route_c_commit": "abc123", "device": "BI-V150"},
            "status": "inconclusive",
            "causes": ["final-isa-unavailable"],
        }
        validate_document(payload)
        self.assertEqual("abc123", document_route_c_commit(payload))

    def test_rejects_unknown_document_type(self):
        with self.assertRaisesRegex(DocumentValidationError, "unsupported"):
            validate_document({"document_type": "large-schema-framework"})

    def test_rejects_missing_route_commit(self):
        payload = valid_experiment()
        payload["environment"].pop("route_c_commit")
        with self.assertRaisesRegex(DocumentValidationError, "route_c_commit"):
            validate_document(payload)


if __name__ == "__main__":
    unittest.main()

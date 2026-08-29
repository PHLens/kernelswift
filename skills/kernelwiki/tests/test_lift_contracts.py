from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

TESTS = Path(__file__).resolve().parent
SKILL_ROOT = TESTS.parent
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from kernelwiki_common import KernelWikiError, canonical_json_bytes, load_yaml_document  # noqa: E402
from lift_schema import validate_lift_document  # noqa: E402


SEALED_HOLDOUT = b"""schema_version: 1
sealed_at: 2026-08-21T00:00:00Z
development_campaigns:
  - kernels/track1-triton/groupedtopk/ascend
  - kernels/track1-triton/flexattention/ascend
  - kernels/track1-triton/mhc_post_layer_mix/ascend
holdout_campaigns:
  - kernels/track1-triton/mm_encoder_attention/ascend
  - kernels/track1-triton/sparse_pooler/ascend
rules:
  - holdout campaigns do not influence outcome mapping or publication defaults
  - holdout campaigns are evaluated only after strict and historical lanes pass tests
  - all historical campaigns remain noncanonical and Designer-only
"""

REQUIRED_ARTIFACTS = (
    "implementation_profile",
    "runtime_snapshot",
    "project_claim",
    "sketch",
    "decision",
    "binding",
    "candidate",
    "report",
    "verdict",
    "team_state",
    "project",
    "base",
    "harness",
)


def valid_identity() -> dict[str, object]:
    return {
        "repository_commit": "b" * 40,
        "skill_tree_sha": "c" * 40,
        "validator_sha256": {"validate_profile": "d" * 64},
        "schema_sha256": {},
    }


def valid_bundle() -> dict[str, object]:
    return {
        "schema_version": 1,
        "proposal_id": "experience-current-round-001",
        "repository_root": "/tmp/repository",
        "project_root": "kernels/example/project",
        "contract_version": 3,
        "loop_contract_identity": valid_identity(),
        "round_id": "round-001",
        "terminal_commit": "a" * 40,
        "terminal_result": "accepted",
        "measurement_exclusive": False,
        "artifacts": {
            name: {
                "name": name,
                "path": f"kernels/example/project/{name}.json",
                "sha256": "e" * 64,
                "required": True,
            }
            for name in REQUIRED_ARTIFACTS
        },
        "canonical_candidate_ref": "kernels/example/project/candidate.py",
        "canonical_report_ref": "kernels/example/project/report.md",
    }


def valid_proposal() -> dict[str, object]:
    return {
        "schema_version": 1,
        "proposal_id": "experience-current-round-001",
        "source_lane": "strict-current-vnext",
        "contract_version": 3,
        "loop_contract_identity": valid_identity(),
        "artifact_hashes": {"candidate": "e" * 64, "report": "f" * 64},
        "terminal": {"round_id": "round-001", "result": "accepted"},
        "scope": {"target_id": "mlu590", "implementation_profile_id": "triton_mlu"},
        "expected": {"comparable": True},
        "observed": [{"kind": "measurement", "value": "improved"}],
        "suggested_publication": {"decision": "include", "mode": "existing-card-example"},
        "transfer_boundaries": ["exact profile and runtime only"],
        "reconsider_when": ["runtime or target changes"],
        "missing_evidence": [],
    }


def valid_review() -> dict[str, object]:
    return {
        "schema_version": 1,
        "proposal_id": "experience-current-round-001",
        "proposal_sha256": "9" * 64,
        "decision": "include",
        "reviewed_by": "kernelwiki-curator",
        "reviewed_at": "2026-08-21T00:00:00Z",
        "rationale": "The scoped evidence teaches a reusable mechanism.",
        "publication_target": {
            "mode": "existing-card-example",
            "card_id": "technique-kernel-fusion",
        },
    }


class LiftContractTests(unittest.TestCase):
    def test_sealed_local_campaign_holdout_bytes_and_membership(self):
        path = SKILL_ROOT / "data" / "local-campaign-holdout.yaml"
        self.assertEqual(SEALED_HOLDOUT, path.read_bytes())
        document = load_yaml_document(path)
        self.assertEqual(
            [
                "kernels/track1-triton/groupedtopk/ascend",
                "kernels/track1-triton/flexattention/ascend",
                "kernels/track1-triton/mhc_post_layer_mix/ascend",
            ],
            document["development_campaigns"],
        )
        self.assertEqual(
            [
                "kernels/track1-triton/mm_encoder_attention/ascend",
                "kernels/track1-triton/sparse_pooler/ascend",
            ],
            document["holdout_campaigns"],
        )

    def test_valid_bundle_proposal_and_review_round_trip(self):
        schema_path = SKILL_ROOT / "data" / "schemas.yaml"
        documents = {
            "terminal_bundle": valid_bundle(),
            "experience_proposal": valid_proposal(),
            "experience_review": valid_review(),
        }
        for kind, document in documents.items():
            with self.subTest(kind=kind):
                validated = validate_lift_document(kind, document, schema_path)
                self.assertEqual(document, validated)
                self.assertEqual(document, json.loads(canonical_json_bytes(validated)))

    def test_grouped_invalid_terminal_bundle_rows(self):
        schema_path = SKILL_ROOT / "data" / "schemas.yaml"

        def unsupported(document):
            document["contract_version"] = 4

        def bad_commit(document):
            document["terminal_commit"] = "not-a-commit"

        def bad_hash(document):
            document["artifacts"]["candidate"]["sha256"] = "bad"

        def bad_path(document):
            document["artifacts"]["candidate"]["path"] = "../outside.py"

        def measurement_exclusive(document):
            document["measurement_exclusive"] = True

        def missing_artifact(document):
            del document["artifacts"]["harness"]

        for label, mutate in (
            ("unsupported contract", unsupported),
            ("bad commit", bad_commit),
            ("bad hash", bad_hash),
            ("escaping artifact path", bad_path),
            ("measurement exclusive", measurement_exclusive),
            ("missing required artifact", missing_artifact),
        ):
            with self.subTest(row=label):
                document = deepcopy(valid_bundle())
                mutate(document)
                with self.assertRaises(KernelWikiError):
                    validate_lift_document("terminal_bundle", document, schema_path)

    def test_grouped_forbidden_proposal_and_incomplete_review_rows(self):
        schema_path = SKILL_ROOT / "data" / "schemas.yaml"

        for forbidden in (
            "next_candidate",
            "recommended_next_change",
            "implementation_instruction",
        ):
            with self.subTest(proposal_field=forbidden):
                proposal = deepcopy(valid_proposal())
                proposal["observed"][0][forbidden] = "do something next"
                with self.assertRaises(KernelWikiError):
                    validate_lift_document("experience_proposal", proposal, schema_path)

        for missing in ("reviewed_by", "rationale", "proposal_sha256"):
            with self.subTest(review_missing=missing):
                review = deepcopy(valid_review())
                del review[missing]
                with self.assertRaises(KernelWikiError):
                    validate_lift_document("experience_review", review, schema_path)


if __name__ == "__main__":
    unittest.main()

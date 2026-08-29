from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

TESTS = Path(__file__).resolve().parent
SKILL_ROOT = TESTS.parent
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from kernelwiki_common import KernelWikiError, canonical_json_bytes, load_yaml_document, sha256_file  # noqa: E402
from lift_schema import validate_lift_document  # noqa: E402
from validate_lift import validate_experience_tree, validate_proposal, validate_review  # noqa: E402


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


def reviewable_proposal() -> dict[str, object]:
    document = valid_proposal()
    document["scope"] = {
        "target_id": "mlu590",
        "implementation_profile_id": "triton_mlu",
        "implementation_profile_version": 1,
        "profile_status": "partial",
        "runtime_fingerprint": "triton 3.6.0 / CoreX 4.4.0",
        "device_architectures": ["mlu590"],
        "language": "python",
        "backend": "triton",
        "shape_signatures": ["M x N"],
        "dtypes": ["fp32"],
        "measurement_fingerprint": "fixture-measurement",
        "comparability": "comparable",
    }
    document["suggested_publication"] = {
        "decision": "include",
        "mode": "existing-card-example",
        "role": "positive",
        "subtype": "performance",
        "card_ids": ["technique-kernel-fusion"],
        "tags": ["performance", "positive"],
    }
    document["transfer_boundaries"] = [
        "target=mlu590",
        "profile=triton_mlu",
        "runtime=triton 3.6.0 / CoreX 4.4.0",
    ]
    return document


def write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document))


def build_review(proposal_path: Path, decision: str) -> dict[str, object]:
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "proposal_id": proposal["proposal_id"],
        "proposal_sha256": sha256_file(proposal_path),
        "decision": decision,
        "reviewed_by": "kernelwiki-curator",
        "reviewed_at": "2026-08-21T00:00:00Z",
        "rationale": "Terminal evidence is complete and the scoped example teaches a reusable mechanism.",
        "publication_target": (
            {"mode": "existing-card-example", "card_id": "technique-kernel-fusion"}
            if decision == "include"
            else None
        ),
    }


def snapshot_publication_trees() -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for name in ("sources", "wiki", "queries", "compiled"):
        root = SKILL_ROOT / name
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            snapshot[path.relative_to(SKILL_ROOT).as_posix()] = path.read_bytes()
    return snapshot


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

    def test_include_defer_exclude_review_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal_path = root / "experience-current-round-001.json"
            write_json(proposal_path, reviewable_proposal())
            proposal = validate_proposal(proposal_path)
            for decision in ("include", "defer", "exclude"):
                with self.subTest(decision=decision):
                    review_path = root / f"review-{decision}.json"
                    write_json(review_path, build_review(proposal_path, decision))
                    validated = validate_review(review_path, proposal)
                    self.assertEqual(decision, validated["decision"])

    def test_invalid_review_and_forbidden_proposal_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal_path = root / "experience-current-round-001.json"
            write_json(proposal_path, reviewable_proposal())
            proposal = validate_proposal(proposal_path)

            def unknown(review):
                review["unexpected"] = True

            def missing_identity(review):
                review["reviewed_by"] = ""

            def missing_rationale(review):
                review["rationale"] = ""

            def stale_hash(review):
                review["proposal_sha256"] = "0" * 64

            def operator_card(review):
                review["publication_target"] = {
                    "mode": "new-general-card",
                    "title": "GroupedTopK optimization",
                    "independent_teaching_value": True,
                }

            def coder_visibility(review):
                review["publication_target"] = {
                    "mode": "existing-card-example",
                    "card_id": "technique-kernel-fusion",
                    "audiences": ["coder"],
                }

            for index, (label, mutate) in enumerate((
                ("unknown field", unknown),
                ("missing identity", missing_identity),
                ("missing rationale", missing_rationale),
                ("stale hash", stale_hash),
                ("operator card", operator_card),
                ("coder visibility", coder_visibility),
            )):
                with self.subTest(row=label):
                    review = build_review(proposal_path, "include")
                    mutate(review)
                    review_path = root / f"invalid-review-{index}.json"
                    write_json(review_path, review)
                    with self.assertRaises(KernelWikiError):
                        validate_review(review_path, proposal)

            poisoned = reviewable_proposal()
            poisoned["expected"]["nested"] = {"implementation_instruction": "change the next kernel"}
            poisoned_path = root / "poisoned.json"
            write_json(poisoned_path, poisoned)
            with self.assertRaises(KernelWikiError):
                validate_proposal(poisoned_path)

    def test_experience_tree_duplicate_and_missing_review_references(self):
        for row in ("missing proposal", "duplicate reviews"):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                candidate_root = root / "candidates" / "experience"
                proposal_path = candidate_root / "experience-current-round-001.json"
                write_json(proposal_path, reviewable_proposal())
                review_root = candidate_root / "reviews"
                if row == "missing proposal":
                    review = build_review(proposal_path, "defer")
                    review["proposal_id"] = "experience-missing"
                    write_json(review_root / "missing.json", review)
                else:
                    review = build_review(proposal_path, "defer")
                    write_json(review_root / "first.json", review)
                    write_json(review_root / "second.json", review)
                with self.assertRaises(KernelWikiError):
                    validate_experience_tree(root)

    def test_validation_never_publishes_and_unreviewed_candidates_are_valid(self):
        before = snapshot_publication_trees()
        result = validate_experience_tree(SKILL_ROOT)
        self.assertEqual({"proposals": 4, "reviews": 4, "included": 3}, result)
        self.assertEqual(before, snapshot_publication_trees())


if __name__ == "__main__":
    unittest.main()

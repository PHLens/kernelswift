import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from validate_profile import (
    ProfileValidationError,
    load_profile,
    require_capability,
    validate_configuration_domain,
    validate_project_claim,
)

FIXTURES = Path(__file__).parent / "fixtures" / "vnext"
PROFILES = FIXTURES / "profiles"
CLAIMS = FIXTURES / "claims"

SNAPSHOT = {
    "target_id": "mlu590",
    "implementation_profile_id": "triton_mlu",
    "triton_version": "3.6.0",
    "device_arch": "mlu-arch",
}


class ValidateProfileTests(unittest.TestCase):
    def test_partial_profile_is_usable_but_unknown_cannot_satisfy_required(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        self.assertEqual("partial", profile["profile_status"])
        with self.assertRaisesRegex(ProfileValidationError, "unproven required capability"):
            require_capability(
                profile,
                "matrix.dot",
                {"dtype": "fp32", "layout": "row_major", "shape": ["M", "N"]},
                "required",
            )

    def test_unknown_required_profile_cannot_satisfy_required(self):
        profile = load_profile(PROFILES / "unknown-required" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "unproven required capability"):
            require_capability(
                profile,
                "memory.load",
                {"dtype": "fp32", "layout": "contiguous", "shape": ["N"]},
                "required",
            )

    def test_supported_and_constrained_satisfy_required(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        self.assertEqual("supported", require_capability(profile, "memory.load", {"dtype": "fp32", "layout": "contiguous", "shape": ["N"]}, "required")["status"])
        self.assertEqual("constrained", require_capability(profile, "memory.store", {"dtype": "fp32", "layout": "contiguous", "shape": ["N"]}, "required")["status"])
        with self.assertRaisesRegex(ProfileValidationError, "unproven required capability"):
            require_capability(profile, "tensor.zeros", {"dtype": "fp32"}, "required")

    def test_profile_declares_all_five_capability_statuses(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        statuses = {entry["status"] for entry in profile["capability_matrix"]}
        self.assertEqual({"supported", "constrained", "unknown", "unsupported", "prohibited"}, statuses)

    def test_c_like_profile_needs_no_triton_symbols(self):
        profile = load_profile(PROFILES / "valid-clike-partial" / "profile.yaml")
        symbols = {entry["implementation_symbol"] for entry in profile["capability_matrix"]}
        self.assertNotIn("tl.dot", symbols)
        self.assertIn("DataCopy", symbols)
        self.assertEqual("fixture-clike-symbols", profile["source_conformance"]["analyzer"])

    def test_profile_requires_all_mandatory_sections(self):
        value = json.loads((PROFILES / "valid-partial" / "profile.yaml").read_text(encoding="utf-8"))
        del value["profiler_evidence"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROFILES / "valid-partial", root / "profile")
            (root / "profile" / "profile.yaml").write_text(json.dumps(value, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "profiler_evidence"):
                load_profile(root / "profile" / "profile.yaml")

    def test_duplicate_capability_ids_are_rejected(self):
        value = json.loads((PROFILES / "valid-partial" / "profile.yaml").read_text(encoding="utf-8"))
        value["capability_matrix"][1]["id"] = value["capability_matrix"][0]["id"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROFILES / "valid-partial", root / "profile")
            (root / "profile" / "profile.yaml").write_text(json.dumps(value, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "duplicate capability"):
                load_profile(root / "profile" / "profile.yaml")

    def test_schema_copy_hash_mismatch_is_rejected(self):
        value = json.loads((PROFILES / "valid-partial" / "profile.yaml").read_text(encoding="utf-8"))
        value["profile_schema_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROFILES / "valid-partial", root / "profile")
            (root / "profile" / "profile.yaml").write_text(json.dumps(value, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "schema-hash-mismatch"):
                load_profile(root / "profile" / "profile.yaml")

    def test_probe_catalog_hash_mismatch_is_rejected(self):
        value = json.loads((PROFILES / "valid-partial" / "profile.yaml").read_text(encoding="utf-8"))
        value["probe_catalog"] = [
            {"probe_id": "fixture-basic-memory-001", "definition_path": "profile.yaml", "definition_sha256": "0" * 64}
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROFILES / "valid-partial", root / "profile")
            (root / "profile" / "profile.yaml").write_text(json.dumps(value, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "definition hash"):
                load_profile(root / "profile" / "profile.yaml")

    def test_valid_claim_validates_against_snapshot(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        result = validate_project_claim(CLAIMS / "valid-claim.json", profile=profile, snapshot=SNAPSHOT)
        self.assertTrue(result["valid"])

    def test_runtime_identity_mismatch_is_environment_blocked(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "environment-blocked"):
            validate_project_claim(
                CLAIMS / "runtime-mismatch.json",
                profile=profile,
                snapshot={**SNAPSHOT, "device_arch": "different-arch"},
            )

    def test_target_id_mismatch_is_environment_blocked(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "environment-blocked"):
            validate_project_claim(
                CLAIMS / "target-id-mismatch.json",
                profile=profile,
                snapshot={**SNAPSHOT, "target_id": "other-target"},
            )

    def test_implementation_profile_mismatch_is_environment_blocked(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "environment-blocked"):
            validate_project_claim(
                CLAIMS / "implementation-profile-mismatch.json",
                profile=profile,
                snapshot={**SNAPSHOT, "implementation_profile_id": "triton_cuda"},
            )

    def test_valid_maintainer_authorized_fallback_disposition(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        result = validate_project_claim(CLAIMS / "valid-fallback-disposition.json", profile=profile, snapshot=SNAPSHOT)
        self.assertEqual(1, len(result["claim"]["qualification_dispositions"]))
        self.assertTrue(result["claim"]["qualification_dispositions"][0]["fallback_authorized"])

    def test_silent_algorithm_substitution_is_rejected(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "silent-substitution"):
            validate_project_claim(CLAIMS / "silent-algorithm-substitution.json", profile=profile, snapshot=SNAPSHOT)

    def test_raw_probe_result_ref_is_rejected(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "raw probe-result reference"):
            validate_project_claim(CLAIMS / "raw-probe-ref-fallback.json", profile=profile, snapshot=SNAPSHOT)

    def test_requirement_hash_mismatch_is_rejected(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        value = json.loads((CLAIMS / "valid-fallback-disposition.json").read_text(encoding="utf-8"))
        value["qualification_dispositions"][0]["requirement"]["fallback_signature"] = {"dtype": "fp16", "axis": "k"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "requirement hash"):
                validate_project_claim(path, profile=profile, snapshot=SNAPSHOT)

    def test_missing_confirmation_is_rejected(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        value = json.loads((CLAIMS / "valid-fallback-disposition.json").read_text(encoding="utf-8"))
        del value["qualification_dispositions"][0]["maintainer_confirmation"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "maintainer confirmation"):
                validate_project_claim(path, profile=profile, snapshot=SNAPSHOT)

    def test_configuration_domain_returns_finite_legal_domain(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        domain = validate_configuration_domain(
            profile,
            [{"name": "num_warps", "values": [1, 2]}, {"name": "num_stages", "values": [2, 3]}],
            {"shape_signature": "project-defined"},
        )
        self.assertEqual(4, len(domain))
        self.assertEqual({"num_warps": 1, "num_stages": 2}, domain[0])

    def test_configuration_domain_prunes_cross_field_exclusions(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        domain = validate_configuration_domain(
            profile,
            [
                {"name": "num_warps", "values": [1, 2]},
                {"name": "num_stages", "values": [2, 3]},
                {"name": "num_ctas", "values": [1, 2]},
            ],
            {"shape_signature": "project-defined"},
        )
        self.assertEqual(0, len(domain))

    def test_configuration_domain_rejects_unknown_legality_value(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "legality"):
            validate_configuration_domain(
                profile,
                [{"name": "num_warps", "values": [1, 8]}],
                {"shape_signature": "project-defined"},
            )

    def test_configuration_domain_rejects_duplicate_values(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "duplicate configuration value"):
            validate_configuration_domain(
                profile,
                [{"name": "num_warps", "values": [1, 1]}],
                {"shape_signature": "project-defined"},
            )

    def test_configuration_domain_rejects_open_ended_range(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "finite value list"):
            validate_configuration_domain(
                profile,
                [{"name": "num_warps", "values": "1..16"}],
                {"shape_signature": "project-defined"},
            )

    def test_configuration_domain_rejects_exact_scope_mismatch(self):
        profile = load_profile(PROFILES / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "exact scope"):
            validate_configuration_domain(
                profile,
                [{"name": "num_warps", "values": [1]}],
                {"shape_signature": "other-project"},
            )

    def test_configuration_domain_blocks_missing_legality(self):
        profile = load_profile(PROFILES / "valid-clike-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "legality-unavailable"):
            validate_configuration_domain(
                profile,
                [{"name": "num_warps", "values": [1]}],
                {"shape_signature": "project-defined"},
            )


if __name__ == "__main__":
    unittest.main()

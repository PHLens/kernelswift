from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from validate_decision import DecisionValidationError, main, validate_decision
from vnext_common import compute_submission_snapshot_id, sha256_canonical_json


FIXTURES = Path(__file__).parent / "fixtures" / "decisions"
VNEXT_FIXTURES = Path(__file__).parent / "fixtures" / "vnext"
DECISION_TEMPLATE = SKILL_ROOT / "references" / "decision-template.md"


class ValidateDecisionTests(unittest.TestCase):
    def assertValidationError(self, text, code):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.md"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(DecisionValidationError) as caught:
                validate_decision(path)
        self.assertEqual(caught.exception.code, code)

    def test_kernel_decision_is_normalized(self):
        result = validate_decision(FIXTURES / "kernel-valid.md")

        self.assertEqual(result["metadata"]["change_scope"], "kernel")
        self.assertEqual(result["metadata"]["change_family"], "kernel-fusion")
        self.assertEqual(result["metadata"]["target_profile"], "triton_mlu")
        self.assertEqual(
            list(result["sketch"]),
            ["D", "O", "C", "H"],
        )

    def test_gcu_decision_profile_and_target_hint_are_supported(self):
        text = (FIXTURES / "kernel-valid.md").read_text(encoding="utf-8")
        text = text.replace('"backend":"mlu"', '"backend":"gcu"', 1)
        text = text.replace('"target_profile":"triton_mlu"', '"target_profile":"triton_gcu"', 1)
        text = text.replace("target=triton_mlu", "target=triton_gcu", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcu-decision.md"
            path.write_text(text, encoding="utf-8")
            result = validate_decision(path, expected_profile="triton_gcu")
        self.assertTrue(result["valid"])
        self.assertEqual(result["metadata"]["backend"], "gcu")
        self.assertEqual(result["sketch"]["H"][0], "target=triton_gcu")

    def test_cuda_decision_profile_and_target_hint_are_supported(self):
        text = (FIXTURES / "kernel-valid.md").read_text(encoding="utf-8")
        text = text.replace('"backend":"mlu"', '"backend":"cuda"', 1)
        text = text.replace('"target_profile":"triton_mlu"', '"target_profile":"triton_cuda"', 1)
        text = text.replace("target=triton_mlu", "target=triton_cuda", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda-decision.md"
            path.write_text(text, encoding="utf-8")
            result = validate_decision(path, expected_profile="triton_cuda")
        self.assertTrue(result["valid"])
        self.assertEqual(result["metadata"]["backend"], "cuda")
        self.assertEqual(result["sketch"]["H"][0], "target=triton_cuda")

    def test_kernel_host_plan_accepts_an_explanatory_reason(self):
        text = (FIXTURES / "kernel-valid.md").read_text(encoding="utf-8")
        text = text.replace("kernel-only change", "no host behavior changes", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.md"
            path.write_text(text, encoding="utf-8")
            result = validate_decision(path)
        self.assertTrue(result["valid"])

    def test_host_decision_is_normalized(self):
        result = validate_decision(FIXTURES / "host-valid.md")

        self.assertEqual(result["metadata"]["change_scope"], "host")
        self.assertIsNone(result["sketch"])
        self.assertEqual(result["host_plan"]["applicability"], "required")

    def test_mixed_decision_is_normalized(self):
        result = validate_decision(FIXTURES / "mixed-valid.md")

        self.assertEqual(result["metadata"]["change_scope"], "mixed")
        self.assertEqual(
            result["metadata"]["change_family"], "mixed-routing-fusion"
        )
        self.assertEqual(result["host_plan"]["applicability"], "required")
        self.assertEqual(result["sketch"]["H"][0], "target=triton_mlu")

    def test_host_change_family_is_normalized(self):
        result = validate_decision(FIXTURES / "host-valid.md")
        self.assertEqual(result["metadata"]["change_family"], "allocation-reuse")

    def test_host_plan_is_required_for_mixed_change(self):
        text = (FIXTURES / "mixed-valid.md").read_text(encoding="utf-8")
        missing_host_plan = text.replace(
            '"applicability":"required"',
            '"applicability":"not-applicable"',
            1,
        )
        self.assertValidationError(missing_host_plan, "host-plan-required")

    def test_expected_profile_must_match_metadata(self):
        with self.assertRaises(DecisionValidationError) as caught:
            validate_decision(
                FIXTURES / "kernel-valid.md",
                expected_profile="triton_cuda",
            )
        self.assertEqual(caught.exception.code, "target-profile-mismatch")

    def test_change_family_is_required_and_slug_shaped(self):
        text = (FIXTURES / "kernel-valid.md").read_text(encoding="utf-8")
        self.assertValidationError(
            text.replace(',"change_family":"kernel-fusion"', "", 1),
            "metadata-field-required",
        )
        self.assertValidationError(
            text.replace("kernel-fusion", "Kernel fusion", 1),
            "metadata-change-family-invalid",
        )

    def test_cli_error_has_path_line_code_and_message(self):
        path = FIXTURES / "kernel-valid.md"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            return_code = main([str(path), "--expected-profile", "wrong-profile"])

        self.assertEqual(return_code, 2)
        self.assertRegex(
            stderr.getvalue(),
            rf"^{re.escape(str(path))}:\d+: target-profile-mismatch: .+\n$",
        )

    def test_hint_directives_must_be_on_separate_lines(self):
        text = (FIXTURES / "kernel-valid.md").read_text(encoding="utf-8")
        two_hints_on_one_line = text.replace(
            "num_warps=1\nnum_stages=2",
            "num_warps=1 num_stages=2",
        )
        self.assertValidationError(
            two_hints_on_one_line,
            "sketch-h-one-directive-per-line",
        )

    def test_kernel_change_requires_sketch_fence(self):
        text = (FIXTURES / "kernel-valid.md").read_text(encoding="utf-8")
        missing_sketch_fence = text.replace("```sketch", "```text", 1)
        self.assertValidationError(missing_sketch_fence, "sketch-fence-missing")

    def test_evaluation_requires_an_observable(self):
        text = (FIXTURES / "kernel-valid.md").read_text(encoding="utf-8")
        missing_observable = text.replace(
            '[{"name":"external_kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}]',
            "[]",
        )
        self.assertValidationError(
            missing_observable,
            "evaluation-observable-required",
        )

    def test_complete_template_examples_validate(self):
        template = DECISION_TEMPLATE.read_text(encoding="utf-8")
        examples = re.findall(r"````markdown\n(# Decision .*?)\n````", template, re.DOTALL)
        self.assertEqual(len(examples), 3)

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "rounds").mkdir()
            (project / "project.md").write_text(
                "# Project\n\n## Runtime Fingerprint\n",
                encoding="utf-8",
            )
            for reference in (
                "baseline_adapter.py",
                "triton_example_001.py",
                "triton_example_003.py",
            ):
                (project / reference).touch()
            for report in ("report_000.md", "report_001.md", "report_003.md"):
                (project / "rounds" / report).touch()

            expected_families = (
                "kernel-fusion",
                "allocation-reuse",
                "no-change",
            )
            for index, (example, expected_family) in enumerate(
                zip(examples, expected_families), start=1
            ):
                path = project / "rounds" / f"decision_example_{index}.md"
                path.write_text(example + "\n", encoding="utf-8")
                result = validate_decision(path, expected_profile="triton_mlu")
                self.assertTrue(result["valid"])
                self.assertEqual(result["metadata"]["change_family"], expected_family)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_vnext_project(root: Path) -> Path:
    """Build a self-contained vNext project root and return the decision path."""
    (root / "rounds").mkdir()
    (root / "state").mkdir()
    (root / "baseline_adapter.py").write_text("class ModelNew: pass\n", encoding="utf-8")
    (root / "rounds" / "report_000.md").write_text("# Report 000\n", encoding="utf-8")
    (root / "project.md").write_text("# Project\n\n## runtime-fingerprint\n\nfixture\n", encoding="utf-8")
    shutil.copyfile(VNEXT_FIXTURES / "sketches" / "valid-kernel.json", root / "rounds" / "sketch_001.json")
    shutil.copytree(VNEXT_FIXTURES / "profiles" / "valid-partial", root / "state" / "implementation_profile_snapshot")
    shutil.copyfile(VNEXT_FIXTURES / "claims" / "valid-claim.json", root / "state" / "project_capability_claim.json")
    decision = root / "rounds" / "decision_001.md"
    text = (VNEXT_FIXTURES / "decisions" / "valid-vnext.md").read_text(encoding="utf-8")
    text = _replace_decision_markers(text, root)
    decision.write_text(text, encoding="utf-8")
    return decision


def _replace_decision_markers(text: str, root: Path) -> str:
    replacements = {
        "__SKETCH_SHA256__": sha256_file(root / "rounds" / "sketch_001.json"),
        "__PROFILE_SHA256__": sha256_file(root / "state" / "implementation_profile_snapshot" / "profile.yaml"),
        "__CLAIM_SHA256__": sha256_file(root / "state" / "project_capability_claim.json"),
    }
    for marker, digest in replacements.items():
        text = text.replace(marker, digest)
    return text


class VNextDecisionTests(unittest.TestCase):
    def test_vnext_decision_requires_existing_hashed_sketch(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = materialize_vnext_project(Path(directory))
            result = validate_decision(decision, project_root=decision.parents[1], expected_implementation_profile="triton_mlu")
            self.assertEqual("rounds/sketch_001.json", result["sketch_ref"])
            self.assertTrue(result["valid"])

    def test_vnext_decision_rejects_wrong_sketch_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_vnext_project(root)
            decision = root / "rounds" / "decision_001.md"
            text = (VNEXT_FIXTURES / "decisions" / "invalid-sketch-hash.md").read_text(encoding="utf-8")
            text = _replace_decision_markers(text, root)
            text = text.replace('"sketch_sha256": "__WRONG_SKETCH_SHA256__"', f'"sketch_sha256": "{ "0" * 64 }"')
            decision.write_text(text, encoding="utf-8")
            # Metadata diverges from the Unified Sketch contract hash.
            with self.assertRaisesRegex(DecisionValidationError, "sketch-contract-mismatch"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_vnext_decision_rejects_missing_reference_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_vnext_project(root)
            (root / "rounds" / "sketch_001.json").unlink()
            with self.assertRaisesRegex(DecisionValidationError, "existing file"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_vnext_decision_requires_runtime_fingerprint_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_vnext_project(root)
            (root / "project.md").write_text("# Project\n\nNo fingerprint here\n", encoding="utf-8")
            with self.assertRaisesRegex(DecisionValidationError, "runtime-fingerprint-anchor"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_snapshot_closure_validates_without_canonical_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_vnext_project(root)
            result = validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")
            self.assertEqual("state/implementation_profile_snapshot/profile.yaml", result["implementation_profile_snapshot_ref"])
            self.assertTrue(result["valid"])

    def test_v2_requires_expected_implementation_profile_only(self):
        with tempfile.TemporaryDirectory() as directory:
            decision = materialize_vnext_project(Path(directory))
            with self.assertRaisesRegex(DecisionValidationError, "schema-v2 requires expected_implementation_profile only"):
                validate_decision(decision, project_root=decision.parents[1])
            with self.assertRaisesRegex(DecisionValidationError, "schema-v2 requires expected_implementation_profile only"):
                validate_decision(decision, expected_profile="triton_mlu", project_root=decision.parents[1], expected_implementation_profile="triton_mlu")

    def test_v1_decision_still_normalizes(self):
        result = validate_decision(FIXTURES / "kernel-valid.md", expected_profile="triton_mlu")
        self.assertEqual(1, result["metadata"]["schema_version"])
        with self.assertRaisesRegex(DecisionValidationError, "schema-v1 uses expected_profile"):
            validate_decision(FIXTURES / "kernel-valid.md", expected_implementation_profile="triton_mlu")

    def test_disconnected_causal_graph_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_vnext_project(root)
            text = decision.read_text(encoding="utf-8")
            replacement = '"causal_graph": {\n    "nodes": ["o.external-kernel-count", "p.wall-time"],\n    "edges": [["o.external-kernel-count", "p.wall-time"]]\n  }'
            text = text.replace(
                '"causal_graph": {\n    "nodes": ["m.reduce-fusion", "o.external-kernel-count", "p.wall-time"],\n    "edges": [\n      ["m.reduce-fusion", "o.external-kernel-count"],\n      ["o.external-kernel-count", "p.wall-time"]\n    ]\n  }',
                replacement,
            )
            decision.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(DecisionValidationError, "not connected"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_unknown_section_is_rejected_in_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_vnext_project(root)
            text = decision.read_text(encoding="utf-8") + "\n## Surprise\n\ncontent\n"
            decision.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(DecisionValidationError, "unknown H2 section"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_fallback_provenance_validates_against_embedded_disposition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_vnext_project(root)
            shutil.copyfile(VNEXT_FIXTURES / "claims" / "valid-fallback-disposition.json", root / "state" / "project_capability_claim.json")
            text = (VNEXT_FIXTURES / "decisions" / "valid-explicit-fallback.md").read_text(encoding="utf-8")
            text = _replace_decision_markers(text, root)
            claim = json.loads((root / "state" / "project_capability_claim.json").read_text(encoding="utf-8"))
            disposition = claim["qualification_dispositions"][0]
            text = text.replace("__DISPOSITION_SHA256__", sha256_canonical_json(disposition))
            text = text.replace('"project_capability_claim_sha256": "__CLAIM_SHA256__"', f'"project_capability_claim_sha256": "{sha256_file(root / "state" / "project_capability_claim.json")}"')
            decision.write_text(text, encoding="utf-8")
            result = validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")
            self.assertEqual("s60-attention-dot-fallback-001", result["fallback_provenance"]["qualification_disposition_id"])
            self.assertTrue(result["fallback_provenance"]["primary_remains_unknown"])

    def test_silent_algorithm_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_vnext_project(root)
            shutil.copyfile(VNEXT_FIXTURES / "claims" / "valid-fallback-disposition.json", root / "state" / "project_capability_claim.json")
            text = (VNEXT_FIXTURES / "decisions" / "invalid-silent-fallback.md").read_text(encoding="utf-8")
            text = _replace_decision_markers(text, root)
            text = text.replace('"project_capability_claim_sha256": "__CLAIM_SHA256__"', f'"project_capability_claim_sha256": "{sha256_file(root / "state" / "project_capability_claim.json")}"')
            decision.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(DecisionValidationError, "fallback provenance"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_disposition_hash_mutation_invalidates_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_vnext_project(root)
            claim_path = root / "state" / "project_capability_claim.json"
            original_claim = json.loads((VNEXT_FIXTURES / "claims" / "valid-fallback-disposition.json").read_text(encoding="utf-8"))
            original_disposition = original_claim["qualification_dispositions"][0]
            # The claim file embeds a MUTATED disposition...
            mutated = json.loads(json.dumps(original_claim))
            mutated["qualification_dispositions"][0]["reason"] = "mutated reason"
            claim_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            # ...but the Decision was written against the ORIGINAL disposition hash.
            text = (VNEXT_FIXTURES / "decisions" / "valid-explicit-fallback.md").read_text(encoding="utf-8")
            text = _replace_decision_markers(text, root)
            text = text.replace("__DISPOSITION_SHA256__", sha256_canonical_json(original_disposition))
            text = text.replace('"project_capability_claim_sha256": "__CLAIM_SHA256__"', f'"project_capability_claim_sha256": "{sha256_file(claim_path)}"')
            decision.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(DecisionValidationError, "disposition"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")


def materialize_final_tuning_project(root: Path) -> tuple[Path, dict[str, str]]:
    """Build a final-tuning project root and return (decision_path, anchors)."""
    (root / "rounds").mkdir()
    (root / "state").mkdir()
    (root / "project.md").write_text("# Project\n\n## runtime-fingerprint\n\nfixture\n", encoding="utf-8")
    (root / "accepted_candidate.py").write_text("class ModelNew: pass\n", encoding="utf-8")
    (root / "base.py").write_text("class Model: pass\n", encoding="utf-8")
    (root / "auto_bench.py").write_text("def main(): pass\n", encoding="utf-8")
    (root / "rounds" / "binding_001.json").write_text('{"schema_version": 1, "round": "001"}\n', encoding="utf-8")
    (root / "state" / "runtime-snapshot.json").write_text('{"interpreter": "python3", "target_id": "mlu590"}\n', encoding="utf-8")
    sketch = json.loads((VNEXT_FIXTURES / "sketches" / "valid-kernel.json").read_text(encoding="utf-8"))
    for hint in sketch["hints"]:
        if hint["name"] == "num_warps":
            hint["modality"] = "preferred"
    (root / "rounds" / "sketch_001.json").write_text(json.dumps(sketch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copytree(VNEXT_FIXTURES / "profiles" / "valid-partial", root / "state" / "implementation_profile_snapshot")
    shutil.copyfile(VNEXT_FIXTURES / "claims" / "valid-claim.json", root / "state" / "project_capability_claim.json")

    anchors = {
        "candidate_sha256": sha256_file(root / "accepted_candidate.py"),
        "binding_sha256": sha256_file(root / "rounds" / "binding_001.json"),
        "sketch_sha256": sha256_file(root / "rounds" / "sketch_001.json"),
        "profile_sha256": sha256_file(root / "state" / "implementation_profile_snapshot" / "profile.yaml"),
        "claim_sha256": sha256_file(root / "state" / "project_capability_claim.json"),
        "runtime_snapshot_sha256": sha256_file(root / "state" / "runtime-snapshot.json"),
        "measurement_fingerprint_sha256": "6" * 64,
        "harness_sha256": sha256_file(root / "auto_bench.py"),
        "base_sha256": sha256_file(root / "base.py"),
    }
    snapshot_id = compute_submission_snapshot_id(anchors)
    return root / "rounds" / "decision_002.md", anchors


def materialize_final_tuning_decision(root: Path, fixture_name: str) -> Path:
    decision_path, anchors = materialize_final_tuning_project(root)
    text = (VNEXT_FIXTURES / "decisions" / fixture_name).read_text(encoding="utf-8")
    replacements = {
        "__SKETCH_SHA256__": anchors["sketch_sha256"],
        "__PROFILE_SHA256__": anchors["profile_sha256"],
        "__CLAIM_SHA256__": anchors["claim_sha256"],
        "__CANDIDATE_SHA256__": anchors["candidate_sha256"],
        "__BINDING_SHA256__": anchors["binding_sha256"],
        "__RUNTIME_SNAPSHOT_SHA256__": anchors["runtime_snapshot_sha256"],
        "__MEASUREMENT_SHA256__": anchors["measurement_fingerprint_sha256"],
        "__HARNESS_SHA256__": anchors["harness_sha256"],
        "__BASE_SHA256__": anchors["base_sha256"],
        "__SNAPSHOT_ID__": compute_submission_snapshot_id(anchors),
    }
    for marker, digest in replacements.items():
        text = text.replace(marker, digest)
    decision_path.write_text(text, encoding="utf-8")
    return decision_path


class VNextFinalTuningDecisionTests(unittest.TestCase):
    def test_valid_final_tuning_decision_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_final_tuning_decision(root, "valid-final-tuning.md")
            result = validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")
            self.assertEqual("final-autotune", result["decision_kind"])
            contract = result["final_tuning_contract"]
            self.assertEqual(3, len(contract["configurations"]))
            self.assertEqual({"num_warps": 1, "num_stages": 2}, contract["fallback_configuration"])
            self.assertTrue(contract["pin_selected_config"])

    def test_final_tuning_semantic_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_final_tuning_decision(root, "invalid-final-tuning-semantic-field.md")
            with self.assertRaisesRegex(DecisionValidationError, "semantic-field"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_final_tuning_profile_domain_violation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_final_tuning_decision(root, "invalid-final-tuning-profile-domain.md")
            with self.assertRaisesRegex(DecisionValidationError, "profile-domain"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_final_tuning_stale_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_final_tuning_decision(root, "valid-final-tuning.md")
            (root / "accepted_candidate.py").write_text("class ModelNew: pass  # changed\n", encoding="utf-8")
            with self.assertRaisesRegex(DecisionValidationError, "anchor"):
                validate_decision(decision, project_root=root, expected_implementation_profile="triton_mlu")

    def test_final_tuning_requires_artifact_index_matching_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = materialize_final_tuning_decision(root, "valid-final-tuning.md")
            renamed = root / "rounds" / "decision_999.md"
            shutil.move(decision, renamed)
            with self.assertRaisesRegex(DecisionValidationError, "artifact_index"):
                validate_decision(renamed, project_root=root, expected_implementation_profile="triton_mlu")


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from validate_binding import BindingValidationError, validate_binding
from validate_profile import load_profile
from validate_sketch import SketchValidationError, validate_sketch

FIXTURES = Path(__file__).parent / "fixtures" / "vnext"
SKETCHES = FIXTURES / "sketches"
BINDINGS = FIXTURES / "bindings"
CANDIDATES = FIXTURES / "candidates"
PROFILES = FIXTURES / "profiles"

VALID_SKETCH = validate_sketch(SKETCHES / "valid-load-store-kernel.json")
VALID_PROFILE = load_profile(PROFILES / "valid-partial" / "profile.yaml")
PROJECT_ROOT = FIXTURES


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(binding_name: str, **kwargs) -> dict:
    return validate_binding(
        BINDINGS / binding_name,
        project_root=kwargs.pop("project_root", PROJECT_ROOT),
        sketch_result=kwargs.pop("sketch_result", VALID_SKETCH),
        profile=kwargs.pop("profile", VALID_PROFILE),
        candidate_path=kwargs.pop("candidate_path", CANDIDATES / "valid_candidate.py"),
        **kwargs,
    )


class ValidateBindingTests(unittest.TestCase):
    def test_valid_many_to_many_binding_covers_every_required_statement(self):
        result = validate("valid-many-to-many.json")
        self.assertEqual({"op.load.row", "op.store.output"}, set(result["coverage"]))
        self.assertEqual("python-ast-triton", result["source_analyzer"])
        self.assertEqual("primitive-call", result["binding_model"])
        self.assertIn("tl.load", result["source_symbols"])

    def test_missing_required_statement_coverage_is_rejected(self):
        with self.assertRaisesRegex(BindingValidationError, "required statement"):
            validate("missing-required-statement.json")

    def test_stale_candidate_hash_is_rejected(self):
        with self.assertRaisesRegex(BindingValidationError, "candidate_sha256"):
            validate("stale-candidate-hash.json")

    def test_declared_primitive_must_be_called_at_the_exact_span(self):
        with self.assertRaisesRegex(BindingValidationError, "exact span"):
            validate("invalid-source-primitive.json")

    def test_candidate_hash_is_validated_against_the_file(self):
        binding = json.loads((BINDINGS / "valid-many-to-many.json").read_text(encoding="utf-8"))
        binding["candidate_sha256"] = sha256_file(CANDIDATES / "final-tuning-pinned.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "candidate_sha256"):
                validate_binding(
                    path,
                    project_root=PROJECT_ROOT,
                    sketch_result=VALID_SKETCH,
                    profile=VALID_PROFILE,
                    candidate_path=CANDIDATES / "valid_candidate.py",
                )

    def test_multi_span_binding_requires_a_reason(self):
        binding = json.loads((BINDINGS / "valid-many-to-many.json").read_text(encoding="utf-8"))
        load_binding = next(item for item in binding["bindings"] if item["statement_id"] == "op.load.row")
        load_binding["source_spans"].append(dict(load_binding["source_spans"][0]))
        load_binding.pop("reason", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "reason"):
                validate_binding(
                    path,
                    project_root=PROJECT_ROOT,
                    sketch_result=VALID_SKETCH,
                    profile=VALID_PROFILE,
                    candidate_path=CANDIDATES / "valid_candidate.py",
                )

    def test_elided_by_requires_reason_and_replacement(self):
        binding = json.loads((BINDINGS / "valid-many-to-many.json").read_text(encoding="utf-8"))
        load_binding = next(item for item in binding["bindings"] if item["statement_id"] == "op.load.row")
        load_binding["relation"] = "elided-by"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "elided-by"):
                validate_binding(
                    path,
                    project_root=PROJECT_ROOT,
                    sketch_result=VALID_SKETCH,
                    profile=VALID_PROFILE,
                    candidate_path=CANDIDATES / "valid_candidate.py",
                )

    def test_required_hint_without_binding_record_is_rejected(self):
        sketch_value = json.loads((SKETCHES / "valid-load-store-kernel.json").read_text(encoding="utf-8"))
        for hint in sketch_value["hints"]:
            if hint["name"] == "num_warps":
                hint["modality"] = "required"
        with tempfile.TemporaryDirectory() as directory:
            sketch_path = Path(directory) / "sketch.json"
            sketch_path.write_text(json.dumps(sketch_value), encoding="utf-8")
            sketch = validate_sketch(sketch_path)
            with self.assertRaisesRegex(BindingValidationError, "required hints have no binding record"):
                validate("valid-many-to-many.json", sketch_result=sketch)

    def test_required_hint_record_must_pass_profile_capability(self):
        sketch_value = json.loads((SKETCHES / "valid-load-store-kernel.json").read_text(encoding="utf-8"))
        for hint in sketch_value["hints"]:
            if hint["name"] == "num_warps":
                hint["modality"] = "required"
        binding = json.loads((BINDINGS / "valid-many-to-many.json").read_text(encoding="utf-8"))
        binding["required_hint_bindings"] = [
            {"name": "num_warps", "contract_name": "resource.num-warps", "signature": {"name": "num_warps"}, "status": "implemented"}
        ]
        with tempfile.TemporaryDirectory() as directory:
            sketch_path = Path(directory) / "sketch.json"
            sketch_path.write_text(json.dumps(sketch_value), encoding="utf-8")
            sketch = validate_sketch(sketch_path)
            binding_path = Path(directory) / "binding.json"
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "unproven required capability"):
                validate_binding(
                    binding_path,
                    project_root=PROJECT_ROOT,
                    sketch_result=sketch,
                    profile=VALID_PROFILE,
                    candidate_path=CANDIDATES / "valid_candidate.py",
                )

    def test_unavailable_declared_analyzer_is_explicitly_unavailable(self):
        clike = load_profile(PROFILES / "valid-clike-partial" / "profile.yaml")
        with self.assertRaisesRegex(BindingValidationError, "source analyzer"):
            validate_binding(
                BINDINGS / "valid-many-to-many.json",
                project_root=PROJECT_ROOT,
                sketch_result=VALID_SKETCH,
                profile=clike,
                candidate_path=CANDIDATES / "valid_candidate.py",
            )

    def test_binding_span_outside_candidate_ownership_is_rejected(self):
        binding = json.loads((BINDINGS / "valid-many-to-many.json").read_text(encoding="utf-8"))
        binding["bindings"][0]["source_spans"][0]["path"] = "auto_bench.py"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "outside candidate ownership"):
                validate_binding(
                    path,
                    project_root=PROJECT_ROOT,
                    sketch_result=VALID_SKETCH,
                    profile=VALID_PROFILE,
                    candidate_path=CANDIDATES / "valid_candidate.py",
                )


class FinalTuningBindingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="binding-final-tuning-"))
        self.accepted = self.root / "accepted_candidate.py"
        self.pinned = self.root / "pinned_candidate.py"
        valid_source = (CANDIDATES / "valid_candidate.py").read_text(encoding="utf-8")
        self.accepted.write_text(valid_source + "    kernel[(1,)](scores, output, token, expert, e, num_warps=1, num_stages=2)\n", encoding="utf-8")
        self.pinned.write_text(valid_source + "    kernel[(1,)](scores, output, token, expert, e, num_warps=2, num_stages=2)\n", encoding="utf-8")
        self.contract = {
            "artifact_index": "002",
            "configurations": [
                {"num_warps": 1, "num_stages": 2},
                {"num_warps": 2, "num_stages": 2},
            ],
        }

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def validate_pinned(self, binding_path: Path, pinned: Path | None = None):
        return validate_binding(
            binding_path,
            project_root=PROJECT_ROOT,
            sketch_result=VALID_SKETCH,
            profile=VALID_PROFILE,
            candidate_path=pinned or (CANDIDATES / "final-tuning-pinned.py"),
            accepted_candidate_path=self.accepted,
            final_tuning_contract=self.contract,
        )

    def test_valid_config_only_pin_passes_with_fresh_binding(self):
        result = self.validate_pinned(BINDINGS / "final-tuning-pinned.json")
        self.assertEqual({"op.load.row", "op.store.output"}, set(result["coverage"]))

    def test_stale_accepted_candidate_binding_is_rejected(self):
        binding = json.loads((BINDINGS / "final-tuning-pinned.json").read_text(encoding="utf-8"))
        binding["candidate_sha256"] = sha256_file(self.accepted)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "candidate_sha256"):
                self.validate_pinned(path)

    def test_pin_introducing_a_new_operation_is_rejected(self):
        modified = self.root / "modified_pinned.py"
        modified.write_text(
            self.pinned.read_text(encoding="utf-8") + "    extra = tl.load(scores, mask=expert < e)\n",
            encoding="utf-8",
        )
        binding = json.loads((BINDINGS / "final-tuning-pinned.json").read_text(encoding="utf-8"))
        binding["candidate_path"] = "modified_pinned.py"
        binding["candidate_sha256"] = sha256_file(modified)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "config"):
                validate_binding(
                    path,
                    project_root=self.root,
                    sketch_result=VALID_SKETCH,
                    profile=VALID_PROFILE,
                    candidate_path=modified,
                    accepted_candidate_path=self.accepted,
                    final_tuning_contract=self.contract,
                )

    def test_undeclared_configuration_change_is_rejected(self):
        modified = self.root / "undeclared_pinned.py"
        modified.write_text(
            (CANDIDATES / "valid_candidate.py").read_text(encoding="utf-8")
            + "    kernel[(1,)](scores, output, token, expert, e, num_warps=1, num_stages=9)\n",
            encoding="utf-8",
        )
        binding = json.loads((BINDINGS / "final-tuning-pinned.json").read_text(encoding="utf-8"))
        binding["candidate_path"] = "undeclared_pinned.py"
        binding["candidate_sha256"] = sha256_file(modified)
        contract = {**self.contract, "configurations": [{"num_warps": 1, "num_stages": 2}, {"num_warps": 2, "num_stages": 2}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "declared"):
                validate_binding(
                    path,
                    project_root=self.root,
                    sketch_result=VALID_SKETCH,
                    profile=VALID_PROFILE,
                    candidate_path=modified,
                    accepted_candidate_path=self.accepted,
                    final_tuning_contract=contract,
                )

    def test_final_tuning_binding_requires_submission_finalization_metadata(self):
        binding = json.loads((BINDINGS / "final-tuning-pinned.json").read_text(encoding="utf-8"))
        binding.pop("artifact_kind")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(binding), encoding="utf-8")
            with self.assertRaisesRegex(BindingValidationError, "submission-finalization"):
                self.validate_pinned(path)


if __name__ == "__main__":
    unittest.main()

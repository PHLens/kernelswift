from contextlib import redirect_stderr
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from validate_decision import DecisionValidationError, main, validate_decision


FIXTURES = Path(__file__).parent / "fixtures" / "decisions"
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


if __name__ == "__main__":
    unittest.main()

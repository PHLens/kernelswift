import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

from make_baseline_adapter import (  # noqa: E402
    BaselineAdapterError,
    find_model_class,
    main as baseline_main,
    make_baseline_adapter,
)
from summarize_trace import (  # noqa: E402
    TraceSummaryError,
    main as trace_main,
    summarize_trace,
)


SOURCE = """\
import math

class Model:
    def __call__(self, value):
        return value + 1

def get_inputs():
    return [1]

def get_init_inputs():
    return []
"""


class BaselineAdapterTests(unittest.TestCase):
    def test_renames_one_top_level_model_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "base.py"
            destination = root / "baseline_adapter.py"
            source.write_text(SOURCE, encoding="utf-8")
            source_before = source.read_bytes()

            make_baseline_adapter(source, destination)

            self.assertEqual(source.read_bytes(), source_before)
            tree = ast.parse(destination.read_text(encoding="utf-8"))
            class_names = [
                node.name for node in tree.body if isinstance(node, ast.ClassDef)
            ]
            self.assertIn("ModelNew", class_names)
            self.assertNotIn("Model", class_names)
            function_names = [
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            ]
            self.assertIn("get_inputs", function_names)
            self.assertIn("get_init_inputs", function_names)

    def test_find_model_class_rejects_zero_or_two_top_level_models(self):
        with self.assertRaisesRegex(
            BaselineAdapterError,
            "expected exactly one top-level Model class, found 0",
        ):
            find_model_class(ast.parse("class Other:\n    pass\n"))

        duplicate = ast.parse(
            "class Model:\n    pass\n\nclass Model:\n    pass\n"
        )
        with self.assertRaisesRegex(
            BaselineAdapterError,
            "expected exactly one top-level Model class, found 2",
        ):
            find_model_class(duplicate)

    def test_existing_destination_requires_cli_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "base.py"
            destination = root / "baseline_adapter.py"
            source.write_text(SOURCE, encoding="utf-8")
            destination.write_text("sentinel\n", encoding="utf-8")

            with self.assertRaisesRegex(
                BaselineAdapterError, "destination already exists"
            ):
                make_baseline_adapter(source, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "sentinel\n")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(baseline_main([str(source), str(destination)]), 2)
            self.assertIn("destination already exists", stderr.getvalue())

            self.assertEqual(
                baseline_main([str(source), str(destination), "--force"]), 0
            )
            tree = ast.parse(destination.read_text(encoding="utf-8"))
            self.assertEqual(
                [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
                ["ModelNew"],
            )


class TraceSummaryTests(unittest.TestCase):
    trace = FIXTURES / "traces" / "scoped-50-calls.json"

    def test_candidate_scope_is_normalized_per_forward_call(self):
        summary = summarize_trace(self.trace, 50, "candidate", 0.1)

        self.assertEqual(summary["device_total_us"], 1000.0)
        self.assertEqual(summary["device_us_per_call"], 20.0)
        self.assertEqual(summary["kernel_count_total"], 50)
        self.assertEqual(summary["kernel_count_per_call"], 1.0)
        self.assertEqual(summary["device_ratio"], 0.2)
        self.assertEqual(
            summary["kernels"],
            [
                {
                    "name": "kernel_a",
                    "count_total": 50,
                    "count_per_call": 1.0,
                    "total_us": 1000.0,
                    "us_per_call": 20.0,
                }
            ],
        )

    def test_reference_scope_is_summarized_independently(self):
        summary = summarize_trace(self.trace, 50, "accepted_reference", None)

        self.assertEqual(summary["device_total_us"], 1500.0)
        self.assertEqual(summary["device_us_per_call"], 30.0)
        self.assertEqual(summary["kernel_count_total"], 50)
        self.assertEqual(summary["kernels"][0]["name"], "reference_kernel")
        self.assertNotIn("device_ratio", summary)

    def test_scope_outside_kernel_is_excluded(self):
        summary = summarize_trace(self.trace, 50, "candidate", None)

        self.assertEqual(summary["kernel_count_total"], 50)
        self.assertNotIn(
            "outside_scope_kernel", [kernel["name"] for kernel in summary["kernels"]]
        )

    def test_missing_scope_and_zero_iterations_are_rejected(self):
        with self.assertRaisesRegex(TraceSummaryError, "scope not found: missing"):
            summarize_trace(self.trace, 50, "missing", None)
        with self.assertRaisesRegex(TraceSummaryError, "iterations must be positive"):
            summarize_trace(self.trace, 0, "candidate", None)

    def test_overlapping_duplicate_scopes_are_rejected(self):
        trace = {
            "traceEvents": [
                {"name": "candidate", "cat": "user_annotation", "ph": "X", "ts": 0, "dur": 20},
                {"name": "candidate", "cat": "user_annotation", "ph": "X", "ts": 10, "dur": 20},
                {"name": "kernel", "cat": "kernel", "ph": "X", "ts": 12, "dur": 1},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            with self.assertRaisesRegex(
                TraceSummaryError, "overlapping scope events: candidate"
            ):
                summarize_trace(path, 1, "candidate", None)

    def test_scope_without_kernel_events_is_rejected(self):
        trace = {
            "traceEvents": [
                {"name": "candidate", "cat": "user_annotation", "ph": "X", "ts": 0, "dur": 20},
                {"name": "kernel", "cat": "kernel", "ph": "X", "ts": 30, "dur": 1},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            with self.assertRaisesRegex(
                TraceSummaryError, "scope has no kernel events: candidate"
            ):
                summarize_trace(path, 1, "candidate", None)

    def test_trace_cli_prints_json_and_reports_actionable_errors(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(
                trace_main(
                    [
                        str(self.trace),
                        "--iterations",
                        "50",
                        "--scope",
                        "candidate",
                        "--wall-ms",
                        "0.1",
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(stdout.getvalue())["device_ratio"], 0.2)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(
                trace_main(
                    [str(self.trace), "--iterations", "50", "--scope", "missing"]
                ),
                2,
            )
        self.assertIn("scope not found: missing", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

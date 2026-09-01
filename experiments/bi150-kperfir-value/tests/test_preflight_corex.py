import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

import preflight_corex as preflight_module
from preflight_corex import (
    _create_stock_vector_add,
    _initialize_device_globals,
    classify_preflight,
    disassembly_succeeded,
    materialize_compiled_artifacts,
)


class PreflightStatusTests(unittest.TestCase):
    def test_all_checks_valid(self):
        status, causes = classify_preflight(
            {
                "source_guard": True,
                "environment": True,
                "compile": True,
                "correctness": True,
                "resources": True,
                "artifacts": True,
                "event_timing": True,
                "disassembly": True,
            }
        )
        self.assertEqual("valid", status)
        self.assertEqual([], causes)

    def test_disassembly_failure_is_inconclusive_not_unsupported(self):
        status, causes = classify_preflight(
            {
                "source_guard": True,
                "environment": True,
                "compile": True,
                "correctness": True,
                "resources": True,
                "artifacts": True,
                "event_timing": True,
                "disassembly": False,
            }
        )
        self.assertEqual("inconclusive", status)
        self.assertEqual(["final-isa-unavailable"], causes)

    def test_source_mismatch_is_invalid(self):
        status, causes = classify_preflight({"source_guard": False})
        self.assertEqual("invalid", status)
        self.assertEqual(["accepted-source-mismatch"], causes)


class ArtifactMaterializationTests(unittest.TestCase):
    def test_materializes_and_hashes_required_artifacts(self):
        compiled = SimpleNamespace(
            asm={
                "ttgir": "module { test.ttgir }\n",
                "llir": "; test llir\n",
                "cubin": b"corex-cubin-bytes",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            result = materialize_compiled_artifacts(compiled, output_dir)
            for key, expected in (
                ("ttgir", b"module { test.ttgir }\n"),
                ("llir", b"; test llir\n"),
                ("cubin", b"corex-cubin-bytes"),
            ):
                record = result[key]
                self.assertEqual("observed", record["status"])
                self.assertEqual(len(expected), record["byte_count"])
                self.assertEqual(hashlib.sha256(expected).hexdigest(), record["sha256"])
                self.assertEqual(
                    expected,
                    (output_dir / record["relative_path"]).read_bytes(),
                )

    def test_missing_artifact_is_unavailable(self):
        compiled = SimpleNamespace(asm={"ttgir": "x", "llir": "y"})
        with tempfile.TemporaryDirectory() as directory:
            result = materialize_compiled_artifacts(compiled, Path(directory))
        self.assertEqual("unavailable", result["cubin"]["status"])
        self.assertEqual("cubin-missing", result["cubin"]["cause"])


class DisassemblyPredicateTests(unittest.TestCase):
    def test_requires_success_code_and_instruction_body(self):
        body = """
0000000000000000 <stock_vector_add>:
       0: 01 02 03 04 MOV R1, R2
"""
        self.assertTrue(disassembly_succeeded(0, body))
        self.assertFalse(disassembly_succeeded(1, body))
        self.assertTrue(
            disassembly_succeeded(
                0,
                "Function : stock_vector_add\n/*0000*/ MOV R1, R2\n",
            )
        )
        self.assertFalse(
            disassembly_succeeded(
                0,
                "/tmp/stock.cubin: file format elf64-iluvatar\n",
            )
        )


class ImportOrderingTests(unittest.TestCase):
    def test_source_guard_precedes_lazy_device_initialization(self):
        source = (EXPERIMENT_ROOT / "scripts" / "preflight_corex.py").read_text(
            encoding="utf-8"
        )
        guard_call = source.index("accepted_sources = source_guard(REPO_ROOT)")
        initializer_call = source.index("    _initialize_device_globals()", guard_call)
        self.assertLess(guard_call, initializer_call)

    def test_lazy_initialization_publishes_jit_function_globals(self):
        captured = {}
        fake_torch = SimpleNamespace(name="torch")
        fake_tl = SimpleNamespace(constexpr=object())
        fake_driver = SimpleNamespace(name="driver")

        def fake_jit(function):
            captured["function"] = function
            return function

        fake_triton = SimpleNamespace(jit=fake_jit)
        fake_runtime = SimpleNamespace(driver=fake_driver)
        modules = {
            "torch": fake_torch,
            "triton": fake_triton,
            "triton.language": fake_tl,
            "triton.runtime": fake_runtime,
        }
        missing = object()
        previous = {
            name: preflight_module.__dict__.get(name, missing)
            for name in ("torch", "triton", "tl", "driver")
        }
        try:
            _initialize_device_globals(modules.__getitem__)
            kernel = _create_stock_vector_add()
            function_globals = captured["function"].__globals__
            self.assertIs(kernel, captured["function"])
            self.assertIs(fake_torch, function_globals["torch"])
            self.assertIs(fake_triton, function_globals["triton"])
            self.assertIs(fake_tl, function_globals["tl"])
            self.assertIs(fake_driver, function_globals["driver"])
        finally:
            for name, value in previous.items():
                if value is missing:
                    preflight_module.__dict__.pop(name, None)
                else:
                    preflight_module.__dict__[name] = value


if __name__ == "__main__":
    unittest.main()

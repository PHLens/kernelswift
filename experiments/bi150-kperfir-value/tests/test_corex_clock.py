import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "lib"))

from corex_clock import (
    ClockHelperBuildError,
    build_corex_clock,
    clock_compile_command,
    validate_clock_helper_ir,
)


VALID_IR = '''
source_filename = "corex_clock.cu"
target triple = "bi-iluvatar-ilurt"

define dso_local noundef i64 @corex_clock64_start() local_unnamed_addr #2 {
entry:
  %clock = tail call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()
  ret i64 %clock
}

define dso_local noundef i64 @corex_clock64_after_u64(i64 noundef %0) #3 {
entry:
  %token.addr = alloca i64, align 8, addrspace(5)
  store i64 %0, ptr addrspace(5) %token.addr, align 8
  %token = load i64, ptr addrspace(5) %token.addr, align 8
  %low.bit = and i64 %token, 1
  %condition = icmp ne i64 %low.bit, 0
  br i1 %condition, label %odd, label %even

odd:
  %odd.clock = call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()
  %encoded = add i64 %odd.clock, 1
  br label %merge

even:
  %even.clock = call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()
  br label %merge

merge:
  %result = phi i64 [ %encoded, %odd ], [ %even.clock, %even ]
  ret i64 %result
}

declare noundef i64 @llvm.nvvm.read.ptx.sreg.clock64() #4

attributes #2 = {
  alwaysinline mustprogress nounwind
}
attributes #3 = { alwaysinline convergent mustprogress nounwind }
attributes #4 = { nocallback nounwind memory(inaccessiblemem: readwrite) }
'''


class ClockCompileCommandTests(unittest.TestCase):
    def test_bitcode_command_targets_ivcore11_device_llvm(self):
        command = clock_compile_command(
            Path("/usr/local/corex-4.4.0"),
            Path("device/corex_clock.cu"),
            Path("out/corex-clock.bc"),
        )
        self.assertEqual(
            "/usr/local/corex-4.4.0/bin/clang++",
            command[0],
        )
        for item in (
            "-x",
            "ivcore",
            "--cuda-device-only",
            "--cuda-gpu-arch=ivcore11",
            "-emit-llvm",
            "-c",
            "-Wno-unused-command-line-argument",
        ):
            self.assertIn(item, command)
        self.assertNotIn("-S", command)
        self.assertIn("-I/usr/local/corex-4.4.0/include", command)

    def test_text_ir_command_uses_emit_llvm_s(self):
        command = clock_compile_command(
            Path("/opt/corex"),
            Path("clock.cu"),
            Path("clock.ll"),
        )
        self.assertIn("-S", command)
        self.assertIn("-emit-llvm", command)
        self.assertNotIn("-c", command)
        self.assertEqual("clock.ll", command[-1])

    def test_rejects_unknown_output_suffix(self):
        with self.assertRaisesRegex(ClockHelperBuildError, "output suffix"):
            clock_compile_command(Path("/opt/corex"), Path("clock.cu"), Path("x.o"))


class ClockHelperIRTests(unittest.TestCase):
    def test_valid_ir_reports_all_required_checks(self):
        checks = validate_clock_helper_ir(VALID_IR)
        self.assertTrue(all(checks.values()))
        self.assertEqual(
            {
                "start_symbol",
                "dependent_end_symbol",
                "start_clock64_intrinsic",
                "end_clock64_intrinsic",
                "end_token_conditional_branch",
                "end_clock_after_branch",
                "end_clocks_in_both_arms",
                "end_encoded_true_arm",
                "end_no_inline_asm",
                "start_alwaysinline",
                "end_alwaysinline",
                "target_triple",
            },
            set(checks),
        )

    def test_missing_dependent_end_symbol_is_rejected(self):
        with self.assertRaisesRegex(ClockHelperBuildError, "dependent_end_symbol"):
            validate_clock_helper_ir(
                VALID_IR.replace("corex_clock64_after_u64", "wrong_end")
            )

    def test_start_body_must_call_clock_intrinsic(self):
        invalid = VALID_IR.replace(
            "%clock = tail call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()",
            "%clock = add i64 1, 2",
            1,
        )
        with self.assertRaisesRegex(ClockHelperBuildError, "start_clock64_intrinsic"):
            validate_clock_helper_ir(invalid)

    def test_end_body_must_call_clock_intrinsic(self):
        invalid = VALID_IR.replace(
            "@llvm.nvvm.read.ptx.sreg.clock64()",
            "@clock.missing()",
        )
        with self.assertRaisesRegex(ClockHelperBuildError, "clock64_intrinsic"):
            validate_clock_helper_ir(invalid)

    def test_end_requires_token_derived_conditional_branch(self):
        invalid = VALID_IR.replace("br i1 %condition", "br i1 false", 1)
        with self.assertRaisesRegex(
            ClockHelperBuildError, "end_token_conditional_branch"
        ):
            validate_clock_helper_ir(invalid)

    def test_end_clocks_must_follow_conditional_branch(self):
        invalid = VALID_IR.replace(
            "  br i1 %condition, label %odd, label %even\n",
            "  %early.clock = call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()\n"
            "  br i1 %condition, label %odd, label %even\n",
            1,
        ).replace(
            "  %odd.clock = call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()\n",
            "  %odd.clock = add i64 %early.clock, 0\n",
            1,
        ).replace(
            "  %even.clock = call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()\n",
            "  %even.clock = add i64 %early.clock, 0\n",
            1,
        )
        with self.assertRaisesRegex(ClockHelperBuildError, "end_clock_after_branch"):
            validate_clock_helper_ir(invalid)

    def test_end_requires_clock_in_both_branch_arms(self):
        invalid = VALID_IR.replace(
            "  %even.clock = call noundef i64 @llvm.nvvm.read.ptx.sreg.clock64()",
            "  %even.clock = add i64 0, 0",
            1,
        )
        with self.assertRaisesRegex(ClockHelperBuildError, "end_clocks_in_both_arms"):
            validate_clock_helper_ir(invalid)

    def test_true_arm_must_encode_clock_plus_one(self):
        invalid = VALID_IR.replace(
            "%encoded = add i64 %odd.clock, 1",
            "%encoded = add i64 %odd.clock, 0",
            1,
        )
        with self.assertRaisesRegex(ClockHelperBuildError, "end_encoded_true_arm"):
            validate_clock_helper_ir(invalid)

    def test_end_rejects_inline_assembly(self):
        invalid = VALID_IR.replace(
            "odd:\n",
            'odd:\n  call void asm sideeffect "", "~{memory}"()\n',
            1,
        )
        with self.assertRaisesRegex(ClockHelperBuildError, "end_no_inline_asm"):
            validate_clock_helper_ir(invalid)

    def test_start_definition_must_retain_alwaysinline_attribute_group(self):
        invalid = VALID_IR.replace(
            "alwaysinline mustprogress nounwind\n}",
            "mustprogress nounwind\n}",
            1,
        )
        with self.assertRaisesRegex(ClockHelperBuildError, "start_alwaysinline"):
            validate_clock_helper_ir(invalid)

    def test_end_definition_must_retain_alwaysinline_attribute_group(self):
        invalid = VALID_IR.replace(
            "alwaysinline convergent mustprogress nounwind",
            "convergent mustprogress nounwind",
            1,
        )
        with self.assertRaisesRegex(ClockHelperBuildError, "end_alwaysinline"):
            validate_clock_helper_ir(invalid)

    def test_inline_alwaysinline_attribute_is_accepted_without_group(self):
        inline_attribute_ir = VALID_IR.replace(
            "local_unnamed_addr #2 {", "local_unnamed_addr alwaysinline {", 1
        ).replace("#3 {", "alwaysinline {", 1)
        checks = validate_clock_helper_ir(inline_attribute_ir)
        self.assertTrue(checks["start_alwaysinline"])
        self.assertTrue(checks["end_alwaysinline"])


class ClockHelperBuildTests(unittest.TestCase):
    def test_builds_in_temporary_paths_then_writes_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corex_root = root / "corex"
            source = root / "corex_clock.cu"
            output_dir = root / "artifacts" / "helper"
            source.write_text("device helper source\n", encoding="utf-8")
            calls = []

            def fake_runner(command, **kwargs):
                calls.append((list(command), kwargs))
                if command[-1] == "--version":
                    return SimpleNamespace(
                        returncode=0,
                        stdout="CoreX clang version 18.0\n",
                        stderr="",
                    )
                output_path = Path(command[command.index("-o") + 1])
                if output_path.suffix == ".bc":
                    output_path.write_bytes(b"test-bitcode")
                else:
                    output_path.write_text(VALID_IR, encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = build_corex_clock(
                corex_root,
                source,
                output_dir,
                runner=fake_runner,
            )

            self.assertEqual("corex-clock-helper", result["document_type"])
            self.assertEqual("valid", result["status"])
            self.assertEqual("ivcore11", result["target"])
            self.assertEqual("bi-iluvatar-ilurt", result["target_triple"])
            self.assertEqual("corex-clock.bc", result["bitcode_path"])
            self.assertEqual("corex-clock.ll", result["ir_path"])
            self.assertTrue(all(result["ir_checks"].values()))
            self.assertEqual(
                result,
                json.loads((output_dir / "clock-helper.json").read_text()),
            )
            self.assertEqual(b"test-bitcode", (output_dir / "corex-clock.bc").read_bytes())
            self.assertEqual(VALID_IR, (output_dir / "corex-clock.ll").read_text())
            self.assertEqual(3, len(calls))

    def test_failed_ir_compile_publishes_no_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "corex_clock.cu"
            output_dir = root / "out"
            source.write_text("source\n", encoding="utf-8")

            def fake_runner(command, **kwargs):
                if command[-1] == "--version":
                    return SimpleNamespace(returncode=0, stdout="clang\n", stderr="")
                output_path = Path(command[command.index("-o") + 1])
                if output_path.suffix == ".bc":
                    output_path.write_bytes(b"partial")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                return SimpleNamespace(returncode=1, stdout="", stderr="compile failed")

            with self.assertRaisesRegex(ClockHelperBuildError, "compile failed"):
                build_corex_clock(
                    root / "corex",
                    source,
                    output_dir,
                    runner=fake_runner,
                )
            self.assertFalse(output_dir.exists())


class BuildScriptContractTests(unittest.TestCase):
    def test_source_guard_precedes_build_and_no_device_imports_exist(self):
        source = (
            EXPERIMENT_ROOT / "scripts" / "build_corex_clock.py"
        ).read_text(encoding="utf-8")
        guard_call = source.index("verify_accepted_sources(REPO_ROOT)")
        build_call = source.index("build_corex_clock(", guard_call)
        self.assertLess(guard_call, build_call)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import triton", source)

    def test_device_helper_has_inline_token_control_without_assembly(self):
        source = (EXPERIMENT_ROOT / "device" / "corex_clock.cu").read_text()
        self.assertIn("__attribute__((always_inline, used))", source)
        self.assertIn("corex_clock64_start()", source)
        self.assertIn(
            "corex_clock64_after_u64(unsigned long long token)", source
        )
        self.assertIn("if (token & 1ULL)", source)
        self.assertIn("return clock64() + 1ULL", source)
        self.assertIn("return clock64();", source)
        self.assertNotIn("asm", source)
        self.assertNotIn("MOV", source)
        self.assertNotIn("TIME", source)


if __name__ == "__main__":
    unittest.main()

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = EXPERIMENT_ROOT / "scripts" / "clock64_probe.py"
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))

import clock64_probe
from clock64_probe import (
    CHAIN_ITERS_EMPTY,
    CHAIN_ITERS_LONG,
    CHAIN_ITERS_SHORT,
    CONTROL_CHAIN_ITERS,
    ExternalClockProbeError,
    NUM_WARPS,
    PROFILE_WORDS,
    classify_external_clock,
    decode_program_slot,
    inspect_linked_llir,
    compile_static_dependency_audit,
    load_clock_helper,
    persist_linked_llir,
    read_worker_payload,
    record_static_dependency_audit,
)


class Clock64StatusTests(unittest.TestCase):
    def valid_checks(self):
        return {
            "linked": True,
            "intrinsic_count_ok": True,
            "writeback_ok": True,
            "positive_deltas": True,
            "short_long_sensitive": True,
            "dependency_verified": True,
            "no_spills": True,
        }

    def test_all_checks_qualify_issue_window(self):
        self.assertEqual(
            ("valid", []),
            classify_external_clock(**self.valid_checks()),
        )

    def test_link_failure_is_inconclusive_not_unsupported(self):
        checks = self.valid_checks()
        checks["linked"] = False
        status, causes = classify_external_clock(**checks)
        self.assertEqual("inconclusive", status)
        self.assertEqual(["external-clock-link-failed"], causes)

    def test_sensitivity_failure_is_inconclusive(self):
        checks = self.valid_checks()
        checks["short_long_sensitive"] = False
        status, causes = classify_external_clock(**checks)
        self.assertEqual("inconclusive", status)
        self.assertEqual(["short-long-sensitivity-failed"], causes)

    def test_multiple_failures_remain_inconclusive(self):
        checks = self.valid_checks()
        checks["writeback_ok"] = False
        checks["dependency_verified"] = False
        status, causes = classify_external_clock(**checks)
        self.assertEqual("inconclusive", status)
        self.assertEqual(
            ["profile-writeback-failed", "end-dependency-unverified"],
            causes,
        )
        self.assertNotEqual("unsupported", status)


class CompactRuntimeErrorTests(unittest.TestCase):
    def test_includes_chained_cause_without_traceback(self):
        class UnsupportedLanguageConstruct(RuntimeError):
            pass

        try:
            try:
                raise UnsupportedLanguageConstruct("unsupported AST node type: Dict")
            except UnsupportedLanguageConstruct as cause:
                raise RuntimeError("Triton compilation failed") from cause
        except RuntimeError as error:
            text = clock64_probe.compact_exception_text(error)

        self.assertIn("RuntimeError: Triton compilation failed", text)
        self.assertIn(
            "UnsupportedLanguageConstruct: unsupported AST node type: Dict", text
        )
        self.assertNotIn("Traceback", text)

    def test_deduplicates_exception_cycles(self):
        error = RuntimeError("outer")
        error.__cause__ = error
        self.assertEqual("RuntimeError: outer", clock64_probe.compact_exception_text(error))


class Clock64SourceContractTests(unittest.TestCase):
    def test_source_guard_runs_before_device_initialization(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        guard = source.index("accepted_sources = source_guard(REPO_ROOT)")
        initialize = source.index("_initialize_device_globals()", guard)
        self.assertLess(guard, initialize)

    def test_direct_external_calls_replace_ptx_inline_assembly(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("core.extern_elementwise", source)
        self.assertIn("corex_clock64_start", source)
        self.assertIn("corex_clock64_after_u64", source)
        self.assertIn('extern_libs={"corex_clock": str(clock_bitcode)}', source)
        self.assertIn("CLOCK_EXTERN_IS_PURE = False", source)
        self.assertGreaterEqual(source.count("is_pure=CLOCK_EXTERN_IS_PURE"), 2)
        self.assertIn("acc = tl.load(seed_ptr).to(tl.uint64)", source)
        self.assertIn("persist_linked_llir", source)
        self.assertNotIn("@core.extern", source)
        self.assertNotIn("%clock64", source)
        self.assertNotIn("inline_asm_elementwise", source)

    def test_jit_clock_helpers_reference_global_dispatch_maps_without_dict_literals(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        helpers = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"read_clock_start", "read_clock_after"}
        }
        self.assertEqual({"read_clock_start", "read_clock_after"}, set(helpers))
        for helper in helpers.values():
            self.assertFalse(
                any(isinstance(node, ast.Dict) for node in ast.walk(helper)),
                f"{helper.name} must not construct a dict inside Triton JIT source",
            )
        self.assertIn("CLOCK_START_DISPATCH", source)
        self.assertIn("CLOCK_AFTER_DISPATCH", source)

    def test_lazy_initialization_constructs_constexpr_dispatch_maps(self):
        fake_core = SimpleNamespace(dtype=lambda name: f"dtype:{name}")
        fake_target = object()
        modules = {
            "torch": SimpleNamespace(),
            "triton": SimpleNamespace(),
            "triton.language": SimpleNamespace(),
            "triton.language.core": fake_core,
            "triton.runtime": SimpleNamespace(driver=SimpleNamespace()),
            "triton.backends.compiler": SimpleNamespace(GPUTarget=fake_target),
        }
        names = (
            "torch",
            "triton",
            "tl",
            "core",
            "driver",
            "GPUTarget",
            "CLOCK_EXTERN_LIB_NAME",
            "CLOCK_EXTERN_LIB_PATH",
            "CLOCK_EXTERN_IS_PURE",
            "CLOCK_START_DISPATCH",
            "CLOCK_AFTER_DISPATCH",
        )
        missing = object()
        previous = {
            name: clock64_probe.__dict__.get(name, missing) for name in names
        }
        try:
            clock64_probe._initialize_device_globals(modules.__getitem__)
            self.assertEqual("", clock64_probe.CLOCK_EXTERN_LIB_NAME)
            self.assertEqual("", clock64_probe.CLOCK_EXTERN_LIB_PATH)
            self.assertIs(False, clock64_probe.CLOCK_EXTERN_IS_PURE)
            self.assertIs(fake_target, clock64_probe.GPUTarget)
            self.assertEqual(
                {(): ("corex_clock64_start", "dtype:uint64")},
                clock64_probe.CLOCK_START_DISPATCH,
            )
            self.assertEqual(
                {
                    ("dtype:uint64",): (
                        "corex_clock64_after_u64",
                        "dtype:uint64",
                    )
                },
                clock64_probe.CLOCK_AFTER_DISPATCH,
            )
            self.assertEqual(
                "constexpr",
                clock64_probe.__annotations__["CLOCK_START_DISPATCH"],
            )
            self.assertEqual(
                "constexpr",
                clock64_probe.__annotations__["CLOCK_AFTER_DISPATCH"],
            )
        finally:
            for name, value in previous.items():
                if value is missing:
                    clock64_probe.__dict__.pop(name, None)
                else:
                    clock64_probe.__dict__[name] = value

    def test_profile_layout_uses_scalar_unmasked_stores(self):
        self.assertEqual(1, NUM_WARPS)
        self.assertEqual(3, PROFILE_WORDS)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("tl.store(output_ptr, acc)", source)
        self.assertIn("tl.store(profile_ptr, generation)", source)
        self.assertIn("tl.store(profile_ptr + 1, start_clock)", source)
        self.assertIn("tl.store(profile_ptr + 2, end_clock)", source)
        self.assertNotIn("writer =", source)
        self.assertNotIn("mask=writer", source)
        self.assertNotIn("block_size: tl.constexpr", source)
        self.assertNotIn("block_size=BLOCK_SIZE", source)
        self.assertGreaterEqual(source.count("num_warps=NUM_WARPS"), 2)
        self.assertNotIn("selected_local_warps", source)
        self.assertNotIn("local_warp", source)

    def test_empty_short_and_long_specializations_are_fixed(self):
        self.assertEqual(0, CHAIN_ITERS_EMPTY)
        self.assertEqual(16, CHAIN_ITERS_SHORT)
        self.assertEqual(256, CHAIN_ITERS_LONG)
        self.assertEqual((0, 16, 256), CONTROL_CHAIN_ITERS)

    def test_static_dependency_audit_precedes_cuda_tensor_allocation(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        guard = source.index("accepted_sources = source_guard(REPO_ROOT)")
        initialize = source.index("_initialize_device_globals()", guard)
        audit = source.index("static_audit = compile_static_dependency_audit(", initialize)
        tensor_allocation = source.index("seed = torch.tensor(", audit)
        self.assertLess(guard, initialize)
        self.assertLess(initialize, audit)
        self.assertLess(audit, tensor_allocation)

    def test_worker_writes_result_before_returning(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        worker = source.index("def _worker_main(")
        write = source.index("write_result(args.output, payload)", worker)
        return_statement = source.index("return 0 if payload", write)
        self.assertLess(write, return_statement)

    def test_module_import_does_not_require_device_libraries(self):
        spec = importlib.util.spec_from_file_location("clock64_probe_fresh", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        self.assertFalse(hasattr(module, "torch"))
        self.assertFalse(hasattr(module, "triton"))
        self.assertFalse(hasattr(module, "tl"))


class ProgramSlotTests(unittest.TestCase):
    def test_decodes_matching_generation_and_wraps_unsigned_delta(self):
        row = decode_program_slot([7, (1 << 64) - 3, 5], generation=7)
        self.assertEqual("observed", row["status"])
        self.assertEqual(8, row["raw_cycle_delta"])
        self.assertEqual(0, row["pid"])

    def test_generation_mismatch_is_unavailable(self):
        row = decode_program_slot([6, 100, 200], generation=7)
        self.assertEqual("unavailable", row["status"])
        self.assertEqual("generation-mismatch", row["cause"])
        self.assertNotIn("raw_cycle_delta", row)

    def test_wrong_slot_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "three words"):
            decode_program_slot([1, 2], generation=1)


class LinkedLLIRTests(unittest.TestCase):
    VALID_LLIR = """
    define void @clock_kernel(ptr %profile, ptr %seed) {
    entry:
      %start = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
      %seed_value = load i64, ptr %seed
      %shifted = shl i64 %seed_value, 13
      %token = xor i64 %shifted, %seed_value
      %bit = and i64 %token, 1
      %condition = icmp ne i64 %bit, 0
      br i1 %condition, label %odd, label %even
    odd:
      %odd.end = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
      %encoded = add i64 %odd.end, 1
      br label %merge
    even:
      %even.end = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
      br label %merge
    merge:
      %end = phi i64 [ %encoded, %odd ], [ %even.end, %even ]
      ret void
    }
    define internal i64 @corex_clock64_start() alwaysinline {
      %clock = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
      ret i64 %clock
    }
    define internal i64 @corex_clock64_after_u64(i64 %token) alwaysinline {
      %bit = and i64 %token, 1
      %condition = icmp ne i64 %bit, 0
      br i1 %condition, label %odd, label %even
    odd:
      %odd.clock = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
      %encoded = add i64 %odd.clock, 1
      ret i64 %encoded
    even:
      %even.clock = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
      ret i64 %even.clock
    }
    """

    def test_accepts_inlined_clocks_after_token_branch(self):
        checks = inspect_linked_llir(self.VALID_LLIR)
        self.assertTrue(checks["linked"])
        self.assertTrue(checks["intrinsic_count_ok"])
        self.assertTrue(checks["dependent_end_branch"])
        self.assertTrue(checks["token_branch_after_start"])
        self.assertEqual(2, checks["end_clock_calls"])
        self.assertTrue(checks["runtime_seed_dependency"])
        self.assertTrue(checks["chain_dependency_verified"])
        self.assertTrue(checks["no_helper_runtime_calls"])
        self.assertTrue(checks["no_inline_asm"])
        self.assertEqual(0, checks["helper_runtime_calls"])
        self.assertEqual(3, checks["clock_intrinsic_calls"])

    def test_rejects_branch_before_start_clock(self):
        llir = self.VALID_LLIR.replace(
            "      %start = call i64 @llvm.nvvm.read.ptx.sreg.clock64()\n",
            "",
            1,
        ).replace(
            "      br i1 %condition, label %odd, label %even\n",
            "      br i1 %condition, label %odd, label %even\n"
            "      %start = call i64 @llvm.nvvm.read.ptx.sreg.clock64()\n",
            1,
        )
        checks = inspect_linked_llir(llir)
        self.assertFalse(checks["token_branch_after_start"])
        self.assertFalse(checks["chain_dependency_verified"])
        self.assertFalse(checks["linked"])

    def test_rejects_unrelated_branch_operand(self):
        llir = self.VALID_LLIR.replace("br i1 %condition", "br i1 %unrelated", 1)
        checks = inspect_linked_llir(llir, minimum_chain_steps=16)
        self.assertTrue(checks["dependent_end_branch"])
        self.assertFalse(checks["runtime_seed_dependency"])
        self.assertFalse(checks["chain_dependency_verified"])

    def test_rejects_retained_helper_runtime_call(self):
        llir = self.VALID_LLIR.replace(
            "      %odd.end = call i64 @llvm.nvvm.read.ptx.sreg.clock64()",
            "      %runtime = call i64 @corex_clock64_start()\n"
            "      %odd.end = call i64 @llvm.nvvm.read.ptx.sreg.clock64()",
            1,
        )
        checks = inspect_linked_llir(llir)
        self.assertEqual(1, checks["helper_runtime_calls"])
        self.assertFalse(checks["no_helper_runtime_calls"])
        self.assertFalse(checks["linked"])

    def test_rejects_inline_assembly_in_diagnostic_function(self):
        llir = self.VALID_LLIR.replace(
            "    odd:\n",
            '    odd:\n      call void asm sideeffect "", "~{memory}"()\n',
            1,
        )
        checks = inspect_linked_llir(llir)
        self.assertEqual(1, checks["inline_asm_calls"])
        self.assertFalse(checks["no_inline_asm"])
        self.assertFalse(checks["linked"])

    def test_unreachable_clock_after_branch_is_not_an_end_clock(self):
        llir = """
        define void @clock_kernel(ptr %seed) {
        entry:
          %start = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          %token = load i64, ptr %seed
          %condition = icmp ne i64 %token, 0
          br i1 %condition, label %done, label %done
        done:
          ret void
        unreachable:
          %end = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          ret void
        }
        """
        checks = inspect_linked_llir(llir)
        self.assertEqual(0, checks["end_clock_calls"])
        self.assertFalse(checks["token_branch_after_start"])
        self.assertFalse(checks["linked"])

    def test_numeric_ssa_reuse_in_unrelated_function_cannot_false_pass(self):
        llir = """
        define void @clock_kernel(ptr %profile) {
        entry:
          %1 = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          %2 = add i64 40, 2
          %3 = icmp ne i64 %2, 0
          br i1 %3, label %odd, label %even
        odd:
          %4 = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          ret void
        even:
          %5 = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          ret void
        }
        define void @unrelated(ptr %seed) {
          %1 = load i64, ptr %seed
          %2 = xor i64 %1, 7
          %3 = icmp ne i64 %2, 0
          ret void
        }
        """
        checks = inspect_linked_llir(llir)
        self.assertTrue(checks["dependent_end_branch"])
        self.assertFalse(checks["runtime_seed_dependency"])
        self.assertFalse(checks["chain_dependency_verified"])

    def test_numeric_ssa_reuse_in_helper_definitions_cannot_false_fail(self):
        llir = """
        define void @clock_kernel(ptr %seed) {
        entry:
          %1 = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          %2 = load i64, ptr %seed
          %3 = shl i64 %2, 13
          %4 = xor i64 %3, %2
          %5 = and i64 %4, 1
          %6 = icmp ne i64 %5, 0
          br i1 %6, label %odd, label %even
        odd:
          %7 = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          ret void
        even:
          %8 = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          ret void
        }
        define internal i64 @corex_clock64_after_u64(i64 %0) alwaysinline {
          %1 = and i64 %0, 1
          %2 = icmp ne i64 %1, 0
          br i1 %2, label %odd, label %even
        odd:
          %3 = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          ret i64 %3
        even:
          %4 = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
          ret i64 %4
        }
        """
        checks = inspect_linked_llir(llir, minimum_chain_steps=2)
        self.assertTrue(checks["dependent_end_branch"])
        self.assertTrue(checks["runtime_seed_dependency"])
        self.assertTrue(checks["chain_dependency_verified"])


class StaticDependencyAuditTests(unittest.TestCase):
    OPTIMIZED_AWAY_LLIR = """
    define void @clock_kernel(ptr %seed) {
    entry:
      %start = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
      %end = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
      %token = load i64, ptr %seed
      %bit = and i64 %token, 1
      %decoded = sub i64 %end, %bit
      ret void
    }
    """

    def test_compile_only_audit_uses_manual_signature_target_and_extern_lib(self):
        captured = {}

        class FakeASTSource:
            def __init__(self, **kwargs):
                captured["source"] = kwargs

        class FakeGPUTarget:
            def __init__(self, backend, arch, warp_size):
                captured["target_args"] = (backend, arch, warp_size)

            def __str__(self):
                return "GPUTarget(backend='cuda', arch=71, warp_size=64)"

        compiled = SimpleNamespace(
            asm={"llir": self.OPTIMIZED_AWAY_LLIR},
            hash="compiled-static-audit",
        )

        def fake_compile(source, *, target, options):
            captured["compile"] = (source, target, options)
            return compiled

        fake_triton = SimpleNamespace(
            compiler=SimpleNamespace(ASTSource=FakeASTSource),
            compile=fake_compile,
        )
        previous_triton = clock64_probe.__dict__.get("triton")
        previous_target = clock64_probe.__dict__.get("GPUTarget")
        try:
            clock64_probe.triton = fake_triton
            clock64_probe.GPUTarget = FakeGPUTarget
            with tempfile.TemporaryDirectory() as directory:
                bitcode = Path(directory) / "corex-clock.bc"
                bitcode.write_bytes(b"bitcode")
                audit = compile_static_dependency_audit(
                    kernel=object(),
                    clock_bitcode=bitcode,
                    artifact_dir=Path(directory) / "audit",
                )
                artifact = audit["artifact"]
                self.assertTrue(Path(artifact["path"]).is_file())
                self.assertEqual(
                    hashlib.sha256(self.OPTIMIZED_AWAY_LLIR.encode()).hexdigest(),
                    artifact["sha256"],
                )
        finally:
            if previous_triton is None:
                clock64_probe.__dict__.pop("triton", None)
            else:
                clock64_probe.triton = previous_triton
            if previous_target is None:
                clock64_probe.__dict__.pop("GPUTarget", None)
            else:
                clock64_probe.GPUTarget = previous_target

        self.assertEqual(
            {
                "seed_ptr": "*i64",
                "profile_ptr": "*i64",
                "output_ptr": "*i64",
                "generation": "i32",
            },
            captured["source"]["signature"],
        )
        self.assertEqual({"chain_iters": 16}, captured["source"]["constants"])
        self.assertEqual(("cuda", 71, 64), captured["target_args"])
        self.assertEqual(1, captured["compile"][2]["num_warps"])
        self.assertEqual(
            str(bitcode.resolve()),
            captured["compile"][2]["extern_libs"]["corex_clock"],
        )
        self.assertEqual("optimized-away", audit["status"])
        self.assertFalse(audit["dependency_verified"])
        self.assertFalse(
            audit["linked_llir_checks"]["chain_dependency_verified"]
        )
        self.assertEqual(2, audit["linked_llir_checks"]["clock_intrinsic_calls"])
        self.assertEqual(0, audit["linked_llir_checks"]["conditional_branch_count"])
        self.assertEqual("compiled-static-audit", audit["compiled_hash"])

    def test_optimized_away_audit_stops_without_cycle_regions(self):
        result = clock64_probe.build_result_skeleton(
            commit="abc123",
            accepted_sources={"auto_bench.py": "hash"},
            helper={
                "bitcode_sha256": "b" * 64,
                "source_sha256": "c" * 64,
                "target": "ivcore11",
                "target_triple": "bi-iluvatar-ilurt",
            },
        )
        may_launch = record_static_dependency_audit(
            result,
            {
                "status": "optimized-away",
                "mode": "compile-only-prelaunch",
                "dependency_verified": False,
                "artifact": {"sha256": "d" * 64},
            },
        )
        self.assertFalse(may_launch)
        self.assertEqual("inconclusive", result["qualification_status"])
        self.assertEqual(
            ["end-dependency-optimized-away"], result["status_causes"]
        )
        self.assertEqual([], result["regions"])
        self.assertEqual(
            "optimized-away", result["static_dependency_audit"]["status"]
        )

    def test_parent_retains_worker_payload_after_nonzero_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker-result.json"
            expected = {
                "document_type": "experiment-result",
                "experiment_status": "inconclusive",
                "status_causes": ["end-dependency-optimized-away"],
            }
            path.write_text(json.dumps(expected), encoding="utf-8")
            payload = read_worker_payload(
                path,
                SimpleNamespace(returncode=134),
            )
        self.assertEqual("inconclusive", payload["experiment_status"])
        self.assertEqual(
            ["end-dependency-optimized-away"], payload["status_causes"]
        )
        self.assertEqual(134, payload["worker_returncode"])


class LinkedLLIRPersistenceTests(unittest.TestCase):
    def test_persists_auditable_llir_with_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            record = persist_linked_llir(Path(directory), "short", "define void @k() {}\n")
            path = Path(record["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
            )
            self.assertEqual("observed", record["status"])


class ClockHelperMetadataTests(unittest.TestCase):
    def test_loads_valid_helper_and_checks_bitcode_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bitcode = root / "corex-clock.bc"
            bitcode.write_bytes(b"clock-bitcode")
            metadata = {
                "document_type": "corex-clock-helper",
                "status": "valid",
                "target": "ivcore11",
                "target_triple": "bi-iluvatar-ilurt",
                "bitcode_sha256": hashlib.sha256(bitcode.read_bytes()).hexdigest(),
                "source_sha256": "a" * 64,
                "ir_checks": {
                    "start_symbol": True,
                    "dependent_end_symbol": True,
                    "start_clock64_intrinsic": True,
                    "end_clock64_intrinsic": True,
                    "end_token_conditional_branch": True,
                    "end_clock_after_branch": True,
                    "end_clocks_in_both_arms": True,
                    "end_encoded_true_arm": True,
                    "end_no_inline_asm": True,
                    "start_alwaysinline": True,
                    "end_alwaysinline": True,
                    "target_triple": True,
                },
            }
            metadata_path = root / "clock-helper.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            loaded = load_clock_helper(bitcode, metadata_path)
            self.assertEqual(metadata["bitcode_sha256"], loaded["bitcode_sha256"])
            self.assertEqual(str(bitcode.resolve()), loaded["bitcode_absolute_path"])

    def test_rejects_bitcode_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bitcode = root / "corex-clock.bc"
            bitcode.write_bytes(b"clock-bitcode")
            metadata_path = root / "clock-helper.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "document_type": "corex-clock-helper",
                        "status": "valid",
                        "target": "ivcore11",
                        "target_triple": "bi-iluvatar-ilurt",
                        "bitcode_sha256": "0" * 64,
                        "source_sha256": "a" * 64,
                        "ir_checks": {"all": True},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ExternalClockProbeError, "bitcode SHA256 mismatch"
            ):
                load_clock_helper(bitcode, metadata_path)


class Clock64MetadataTests(unittest.TestCase):
    def test_result_metadata_keeps_program_issue_window_semantics(self):
        payload = clock64_probe.build_result_skeleton(
            commit="abc123",
            accepted_sources={"auto_bench.py": "hash"},
            helper={
                "bitcode_sha256": "b" * 64,
                "source_sha256": "c" * 64,
                "target": "ivcore11",
                "target_triple": "bi-iluvatar-ilurt",
            },
        )
        self.assertEqual("experiment-result", payload["document_type"])
        self.assertEqual(
            "issue-window", payload["instrumentation"]["measurement_semantics"]
        )
        self.assertEqual([0], payload["instrumentation"]["selected_pids"])
        self.assertEqual(3, payload["instrumentation"]["profile_words"])
        self.assertEqual(
            "token-dependent-control-dependency",
            payload["instrumentation"]["completion_dependency"],
        )
        self.assertNotIn("selected_local_warps", payload["instrumentation"])
        self.assertEqual([0, 16, 256], payload["chain_iters"])
        self.assertIn("final-isa-unavailable", payload["limitations"])
        self.assertIn(
            "noinline-helper-runtime-hang-observed", payload["limitations"]
        )
        self.assertIn(
            "post-noinline-gpu-context-unavailable", payload["limitations"]
        )


if __name__ == "__main__":
    unittest.main()

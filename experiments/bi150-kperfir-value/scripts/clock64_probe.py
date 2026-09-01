#!/usr/bin/env python3
"""Qualify program-level BI150 issue windows through a CoreX clock helper.

Accepted source hashes and the Task-1 helper are verified before Torch or
Triton initialize. The helper is linked as LLVM bitcode through ``extern_libs``;
no target assembly spelling or compiler-lowering modification is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "lib"))

from profile_buffer import summarize_cycles, unsigned_cycle_delta
from result_contract import validate_document
from source_guard import SourceGuardError, sha256_file, verify_accepted_sources

CHAIN_ITERS_EMPTY = 0
CHAIN_ITERS_SHORT = 16
CHAIN_ITERS_LONG = 256
CONTROL_CHAIN_ITERS = (
    CHAIN_ITERS_EMPTY,
    CHAIN_ITERS_SHORT,
    CHAIN_ITERS_LONG,
)
CONTROL_LABELS = {
    CHAIN_ITERS_EMPTY: "empty",
    CHAIN_ITERS_SHORT: "short",
    CHAIN_ITERS_LONG: "long",
}
NUM_WARPS = 1
WARP_SIZE = 64
PROFILE_WORDS = 3
OUTPUT_WORDS = 1
RUNTIME_SEED = 0x0123456789ABCDEF
MASK64 = (1 << 64) - 1
GENERATION = 1
WARMUP_RUNS = 20
DEFAULT_SAMPLES = 50
STATIC_AUDIT_CHAIN_ITERS = CHAIN_ITERS_SHORT
STATIC_AUDIT_SIGNATURE = {
    "seed_ptr": "*i64",
    "profile_ptr": "*i64",
    "output_ptr": "*i64",
    "generation": "i32",
}
STATIC_AUDIT_TARGET = ("cuda", 71, 64)
DEFAULT_HELPER_DIR = EXPERIMENT_ROOT / "artifacts" / "external-clock" / "helper"
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT
    / "artifacts"
    / "external-clock"
    / "synthetic"
    / "result.json"
)

# Triton's code generator permits globals annotated exactly as ``constexpr``.
# The maps are populated only after the accepted-source guard and lazy device
# imports, so importing this module locally still requires no Torch/Triton.
CLOCK_EXTERN_LIB_NAME: constexpr
CLOCK_EXTERN_LIB_PATH: constexpr
CLOCK_EXTERN_IS_PURE: constexpr
CLOCK_START_DISPATCH: constexpr
CLOCK_AFTER_DISPATCH: constexpr


class ExternalClockProbeError(RuntimeError):
    """Raised when local evidence is malformed before or during the probe."""


def compact_exception_text(error: BaseException, max_chars: int = 4000) -> str:
    """Return a compact exception chain without rendering a traceback."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = " ".join(str(current).split()) or "<no message>"
        if len(message) > 1200:
            message = message[:1197] + "..."
        parts.append(f"{type(current).__name__}: {message}")
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return " <- caused by: ".join(parts)[:max_chars]


def current_commit(repo_root: Path) -> str:
    configured = os.environ.get("ROUTE_C_COMMIT", "").strip()
    if configured:
        return configured
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def classify_external_clock(
    *,
    linked: bool,
    intrinsic_count_ok: bool,
    writeback_ok: bool,
    positive_deltas: bool,
    short_long_sensitive: bool,
    dependency_verified: bool,
    no_spills: bool,
) -> tuple[str, list[str]]:
    """Classify qualification without claiming that BI150 lacks a clock."""
    causes: list[str] = []
    checks = (
        (linked, "external-clock-link-failed"),
        (intrinsic_count_ok, "clock-intrinsic-count-invalid"),
        (writeback_ok, "profile-writeback-failed"),
        (positive_deltas, "nonpositive-cycle-delta"),
        (short_long_sensitive, "short-long-sensitivity-failed"),
        (dependency_verified, "end-dependency-unverified"),
        (no_spills, "clock-helper-introduced-spill"),
    )
    for passed, cause in checks:
        if not passed:
            causes.append(cause)
    return ("valid", []) if not causes else ("inconclusive", causes)


def classify_runtime_failure(error_text: str) -> str:
    lowered = error_text.lower()
    if any(
        marker in lowered
        for marker in (
            "link extern",
            "linking external",
            "failed to parse library",
            "undefined symbol",
            "corex_clock64_start",
            "corex_clock64_after_u64",
            "extern_libs",
        )
    ):
        return "external-clock-link-failed"
    return "external-clock-probe-failed"


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalClockProbeError(f"{label} must be an object")
    return value


def load_clock_helper(bitcode_path: Path, metadata_path: Path) -> dict[str, Any]:
    """Load and bind Task-1 metadata to the exact local bitcode bytes."""
    bitcode = Path(bitcode_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    if not bitcode.is_file():
        raise ExternalClockProbeError(f"clock bitcode is missing: {bitcode}")
    if not metadata_file.is_file():
        raise ExternalClockProbeError(f"clock helper metadata is missing: {metadata_file}")

    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata = dict(_require_mapping(payload, "clock helper metadata"))
    if metadata.get("document_type") != "corex-clock-helper":
        raise ExternalClockProbeError("clock helper document_type is invalid")
    if metadata.get("status") != "valid":
        raise ExternalClockProbeError("clock helper status is not valid")
    if metadata.get("target") != "ivcore11":
        raise ExternalClockProbeError("clock helper target must be ivcore11")
    if metadata.get("target_triple") != "bi-iluvatar-ilurt":
        raise ExternalClockProbeError(
            "clock helper target triple must be bi-iluvatar-ilurt"
        )
    expected_hash = metadata.get("bitcode_sha256")
    actual_hash = sha256_file(bitcode)
    if expected_hash != actual_hash:
        raise ExternalClockProbeError(
            "clock bitcode SHA256 mismatch: "
            f"expected {expected_hash}, observed {actual_hash}"
        )
    source_hash = metadata.get("source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ExternalClockProbeError("clock helper source_sha256 is invalid")
    ir_checks = _require_mapping(metadata.get("ir_checks"), "clock helper ir_checks")
    if not ir_checks or not all(value is True for value in ir_checks.values()):
        raise ExternalClockProbeError("clock helper IR checks are not all valid")

    metadata["bitcode_absolute_path"] = str(bitcode)
    metadata["metadata_sha256"] = sha256_file(metadata_file)
    return metadata


def _helper_identity(helper: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target": helper.get("target", "unknown"),
        "target_triple": helper.get("target_triple", "unknown"),
        "source_sha256": helper.get("source_sha256", "unknown"),
        "bitcode_sha256": helper.get("bitcode_sha256", "unknown"),
        "metadata_sha256": helper.get("metadata_sha256", "unknown"),
        "symbols": ["corex_clock64_start", "corex_clock64_after_u64"],
    }


def build_result_skeleton(
    *,
    commit: str,
    accepted_sources: dict[str, str],
    helper: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "document_type": "experiment-result",
        "experiment_id": "bi150-external-clock-qualification",
        "environment": {"route_c_commit": commit},
        "variant": {
            "kernel_variant": "synthetic-external-clock-one-program",
            "num_warps": NUM_WARPS,
            "execution_mode": "eager",
        },
        "source": {
            "accepted_source_hashes": accepted_sources,
            "diagnostic_sha256": sha256_file(Path(__file__)),
            "clock_helper": _helper_identity(helper),
        },
        "instrumentation": {
            "mode": "corex-external-bitcode-inline-control",
            "completion_dependency": "token-dependent-control-dependency",
            "region_id": "dependency-chain",
            "selected_pids": [0],
            "time_unit": "raw-cycle",
            "storage": "one-program-generation-start-end",
            "measurement_semantics": "issue-window",
            "profile_words": PROFILE_WORDS,
            "clock_symbols": [
                "corex_clock64_start",
                "corex_clock64_after_u64",
            ],
        },
        "chain_iters": list(CONTROL_CHAIN_ITERS),
        "warmup_runs": WARMUP_RUNS,
        "regions": [],
        "qualification_status": "inconclusive",
        "status_causes": [],
        "experiment_status": "inconclusive",
        "limitations": [
            "program-level-only",
            "issue-window-not-execution-duration",
            "final-isa-unavailable",
            "noinline-helper-runtime-hang-observed",
            "post-noinline-gpu-context-unavailable",
        ],
    }


def _initialize_device_globals(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> None:
    """Publish device libraries and constexpr dispatch maps after verification."""
    global torch, triton, tl, core, driver, GPUTarget
    global CLOCK_EXTERN_LIB_NAME, CLOCK_EXTERN_LIB_PATH, CLOCK_EXTERN_IS_PURE
    global CLOCK_START_DISPATCH, CLOCK_AFTER_DISPATCH
    torch = import_module("torch")
    triton = import_module("triton")
    tl = import_module("triton.language")
    core = import_module("triton.language.core")
    driver = import_module("triton.runtime").driver
    GPUTarget = import_module("triton.backends.compiler").GPUTarget
    CLOCK_EXTERN_LIB_NAME = ""
    CLOCK_EXTERN_LIB_PATH = ""
    CLOCK_EXTERN_IS_PURE = False
    uint64 = core.dtype("uint64")
    CLOCK_START_DISPATCH = {(): ("corex_clock64_start", uint64)}
    CLOCK_AFTER_DISPATCH = {
        (uint64,): ("corex_clock64_after_u64", uint64)
    }


def _create_external_clock_kernel() -> Any:
    """Create JIT helpers that call Triton's builtin extern op directly."""
    global read_clock_start, read_clock_after

    @triton.jit
    def read_clock_start():
        return core.extern_elementwise(
            CLOCK_EXTERN_LIB_NAME,
            CLOCK_EXTERN_LIB_PATH,
            [],
            CLOCK_START_DISPATCH,
            is_pure=CLOCK_EXTERN_IS_PURE,
        )

    @triton.jit
    def read_clock_after(token):
        encoded = core.extern_elementwise(
            CLOCK_EXTERN_LIB_NAME,
            CLOCK_EXTERN_LIB_PATH,
            [token],
            CLOCK_AFTER_DISPATCH,
            is_pure=CLOCK_EXTERN_IS_PURE,
        )
        return encoded - (token & 1)

    @triton.jit
    def external_clock_kernel(
        seed_ptr,
        profile_ptr,
        output_ptr,
        generation,
        chain_iters: tl.constexpr,
    ):
        acc = tl.load(seed_ptr).to(tl.uint64)

        start_clock = read_clock_start()
        for _ in tl.static_range(0, chain_iters):
            acc = acc ^ (acc << 13)
            acc = acc + 0x5DEECE66D
            acc = acc ^ (acc >> 7)
        token = acc
        end_clock = read_clock_after(token)

        tl.store(output_ptr, acc)
        tl.store(profile_ptr, generation)
        tl.store(profile_ptr + 1, start_clock)
        tl.store(profile_ptr + 2, end_clock)

    return external_clock_kernel


def _expected_output(chain_iters: int, seed: int = RUNTIME_SEED) -> int:
    value = seed & MASK64
    for _ in range(chain_iters):
        value ^= (value << 13) & MASK64
        value = (value + 0x5DEECE66D) & MASK64
        value ^= value >> 7
        value &= MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


def decode_program_slot(values: Sequence[int], generation: int) -> dict[str, Any]:
    if len(values) != PROFILE_WORDS:
        raise ValueError("program profile slot must contain exactly three words")
    observed_generation, start, end = (int(value) for value in values)
    row: dict[str, Any] = {
        "pid": 0,
        "start_boundary": "chain-start",
        "end_boundary": "chain-end",
    }
    if observed_generation != generation:
        row.update(status="unavailable", cause="generation-mismatch")
        return row
    row.update(
        status="observed",
        cause="none",
        raw_cycle_start=start,
        raw_cycle_end=end,
        raw_cycle_delta=unsigned_cycle_delta(start, end, 64),
    )
    return row


_SSA_VALUE_RE = re.compile(r"%[-a-zA-Z$._0-9]+")


def _ssa_definitions(llir: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for line in llir.splitlines():
        match = re.match(r"\s*(%[-a-zA-Z$._0-9]+)\s*=\s*(.*)", line)
        if match:
            definitions[match.group(1)] = match.group(2)
    return definitions


def _trace_ssa(definitions: Mapping[str, str], root: str) -> list[str]:
    traced: list[str] = []
    pending = [root]
    visited: set[str] = set()
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        definition = definitions.get(value)
        if definition is None:
            continue
        traced.append(definition)
        pending.extend(_SSA_VALUE_RE.findall(definition))
    return traced


def _llvm_function_bodies(llir: str) -> list[str]:
    """Return complete LLVM function definitions without mixing SSA scopes."""
    starts = [match.start() for match in re.finditer(r"(?m)^\s*define\b", llir)]
    functions: list[str] = []
    for index, start in enumerate(starts):
        boundary = starts[index + 1] if index + 1 < len(starts) else len(llir)
        candidate = llir[start:boundary]
        opening_brace = candidate.find("{")
        if opening_brace < 0:
            continue
        closing_brace = re.search(
            r"(?m)^\s*}\s*(?:;[^\n]*)?$",
            candidate[opening_brace + 1 :],
        )
        if closing_brace is None:
            continue
        end = opening_brace + 1 + closing_brace.end()
        functions.append(candidate[:end])
    return functions


def _conditional_branches(function: str) -> list[tuple[int, str, str, str]]:
    """Return conditional branch positions, operands, and successor labels."""
    branches: list[tuple[int, str, str, str]] = []
    pattern = re.compile(
        r"(?m)^\s*br\s+i1\s+(%[-a-zA-Z$._0-9]+)\s*,\s*"
        r"label\s+%([-a-zA-Z$._0-9]+)\s*,\s*"
        r"label\s+%([-a-zA-Z$._0-9]+)"
    )
    for match in pattern.finditer(function):
        branches.append(
            (match.start(), match.group(1), match.group(2), match.group(3))
        )
    return branches


def _llvm_basic_blocks(function: str) -> dict[str, str]:
    """Split one LLVM function into named basic blocks."""
    opening = function.find("{")
    body = function[opening + 1 :] if opening >= 0 else function
    matches = list(
        re.finditer(r"(?m)^\s*([-a-zA-Z$._0-9]+):(?:\s*;.*)?$", body)
    )
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        blocks[match.group(1)] = body[match.end() : end]
    return blocks


def _reachable_clock_count(
    function: str,
    true_label: str,
    false_label: str,
) -> int:
    """Count end-clock calls in branch successors and their merge blocks."""
    blocks = _llvm_basic_blocks(function)
    pending = [true_label, false_label]
    visited: set[str] = set()
    clock_count = 0
    while pending:
        label = pending.pop()
        if label in visited:
            continue
        visited.add(label)
        block = blocks.get(label)
        if block is None:
            continue
        clock_count += len(
            re.findall(
                r"(?m)\bcall\b[^\n]*@llvm\.nvvm\.read\.ptx\.sreg\.clock64\s*\(",
                block,
            )
        )
        pending.extend(
            re.findall(r"\blabel\s+%([-a-zA-Z$._0-9]+)", block)
        )
    return clock_count


def _inlined_clock_function(
    llir: str,
) -> tuple[str, list[re.Match], list[tuple[int, str, str, str]]] | None:
    """Find the diagnostic function containing inlined clocks and a branch."""
    candidates: list[
        tuple[str, list[re.Match], list[tuple[int, str, str, str]]]
    ] = []
    ordered: list[
        tuple[str, list[re.Match], list[tuple[int, str, str, str]]]
    ] = []
    for function in _llvm_function_bodies(llir):
        clock_calls = list(
            re.finditer(
                r"(?m)\bcall\b[^\n]*@llvm\.nvvm\.read\.ptx\.sreg\.clock64\s*\(",
                function,
            )
        )
        branches = _conditional_branches(function)
        if len(clock_calls) < 2:
            continue
        record = (function, clock_calls, branches)
        candidates.append(record)
        if any(
            clock_calls[0].start() < branch[0]
            and any(clock.start() > branch[0] for clock in clock_calls[1:])
            for branch in branches
        ):
            ordered.append(record)
    return (ordered or candidates)[-1] if candidates else None


def inspect_linked_llir(
    llir: str,
    *,
    minimum_chain_steps: int = 0,
) -> dict[str, Any]:
    """Trace an inlined token branch from the runtime chain to end clocks."""
    retained = _inlined_clock_function(llir)
    if retained is None:
        retained_function = ""
        clock_calls: list[re.Match] = []
        branches: list[tuple[int, str, str, str]] = []
    else:
        retained_function, clock_calls, branches = retained

    definitions = _ssa_definitions(retained_function)
    branch_records: list[dict[str, Any]] = []
    first_clock_position = clock_calls[0].start() if clock_calls else -1
    for position, operand, true_label, false_label in branches:
        textual_end_clocks = [clock for clock in clock_calls if clock.start() > position]
        reachable_end_clocks = _reachable_clock_count(
            retained_function,
            true_label,
            false_label,
        )
        traced = _trace_ssa(definitions, operand)
        runtime_seed_load = any(
            re.search(r"\bload\b[^\n]*\bi64\b", definition)
            for definition in traced
        )
        chain_operation_count = sum(
            bool(re.search(r"\b(?:xor|add|shl|lshr)\b", definition))
            for definition in traced
        )
        branch_records.append(
            {
                "position": position,
                "operand": operand,
                "true_label": true_label,
                "false_label": false_label,
                "after_start": first_clock_position >= 0
                and first_clock_position < position,
                "end_clock_calls": min(
                    len(textual_end_clocks), reachable_end_clocks
                ),
                "reachable_end_clock_calls": reachable_end_clocks,
                "runtime_seed_load": runtime_seed_load,
                "chain_operation_count": chain_operation_count,
            }
        )

    eligible = [
        record
        for record in branch_records
        if record["after_start"] and record["end_clock_calls"] >= 1
    ]
    selected = max(
        eligible,
        key=lambda record: (
            bool(record["runtime_seed_load"]),
            int(record["chain_operation_count"]),
        ),
        default=None,
    )
    required_operations = 0 if minimum_chain_steps == 0 else min(minimum_chain_steps, 4)
    token_branch_after_start = selected is not None
    runtime_seed_load = bool(selected and selected["runtime_seed_load"])
    chain_operation_count = int(selected["chain_operation_count"]) if selected else 0
    end_clock_calls = int(selected["end_clock_calls"]) if selected else 0
    chain_dependency_verified = bool(
        selected
        and runtime_seed_load
        and chain_operation_count >= required_operations
    )
    helper_definitions = bool(
        re.search(r"(?m)^\s*define\b[^\n]*@corex_clock64_start\s*\(", llir)
        or re.search(
            r"(?m)^\s*define\b[^\n]*@corex_clock64_after_u64\s*\(", llir
        )
    )
    helper_runtime_calls = len(
        re.findall(
            r"(?m)\bcall\b[^\n]*@corex_clock64_(?:start|after_u64)\s*\(",
            llir,
        )
    )
    no_helper_runtime_calls = helper_runtime_calls == 0
    inline_asm_calls = len(re.findall(r"(?m)\bcall\b[^\n]*\basm\b", retained_function))
    no_inline_asm = inline_asm_calls == 0
    intrinsic_count_ok = len(clock_calls) >= 2
    linked = bool(
        retained_function
        and intrinsic_count_ok
        and token_branch_after_start
        and end_clock_calls >= 1
        and no_helper_runtime_calls
        and no_inline_asm
    )
    start_before_end = bool(token_branch_after_start and end_clock_calls >= 1)
    return {
        "linked": linked,
        "helper_definitions_retained": helper_definitions,
        "helper_runtime_calls": helper_runtime_calls,
        "no_helper_runtime_calls": no_helper_runtime_calls,
        "inline_asm_calls": inline_asm_calls,
        "no_inline_asm": no_inline_asm,
        "clock_intrinsic_calls": len(clock_calls),
        "intrinsic_count_ok": intrinsic_count_ok,
        "conditional_branch_count": len(branches),
        "dependent_end_branch": bool(branches),
        "token_branch_after_start": token_branch_after_start,
        "end_clock_calls": end_clock_calls,
        "runtime_seed_dependency": runtime_seed_load,
        "chain_operation_count": chain_operation_count,
        "required_chain_operations": required_operations,
        "chain_dependency_verified": chain_dependency_verified,
        "start_before_end": start_before_end,
    }


def _evidence_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(EXPERIMENT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def persist_linked_llir(artifact_dir: Path, label: str, llir: str) -> dict[str, Any]:
    destination = Path(artifact_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"clock64-{label}.ll"
    temporary = destination / f".{path.name}.tmp"
    temporary.write_text(llir, encoding="utf-8")
    os.replace(temporary, path)
    data = path.read_bytes()
    return {
        "status": "observed",
        "path": _evidence_path(path),
        "byte_count": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def compile_static_dependency_audit(
    *,
    kernel: Any,
    clock_bitcode: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Compile one short-chain specialization without allocating GPU tensors."""
    bitcode = Path(clock_bitcode).resolve()
    source = triton.compiler.ASTSource(
        fn=kernel,
        signature=dict(STATIC_AUDIT_SIGNATURE),
        constants={"chain_iters": STATIC_AUDIT_CHAIN_ITERS},
    )
    target = GPUTarget(*STATIC_AUDIT_TARGET)
    options = {
        "num_warps": NUM_WARPS,
        "extern_libs": {"corex_clock": str(bitcode)},
    }
    compiled = triton.compile(source, target=target, options=options)
    llir = compiled.asm.get("llir", "")
    if not isinstance(llir, str):
        llir = bytes(llir).decode("utf-8", "replace")
    if not llir:
        raise ExternalClockProbeError("compile-only dependency audit produced no linked LLIR")
    checks = inspect_linked_llir(
        llir,
        minimum_chain_steps=STATIC_AUDIT_CHAIN_ITERS,
    )
    artifact = persist_linked_llir(
        artifact_dir,
        "static-dependency-audit",
        llir,
    )
    dependency_verified = bool(checks["chain_dependency_verified"])
    return {
        "status": "verified" if dependency_verified else "optimized-away",
        "mode": "compile-only-prelaunch",
        "chain_iters": STATIC_AUDIT_CHAIN_ITERS,
        "signature": dict(STATIC_AUDIT_SIGNATURE),
        "target": str(target),
        "extern_libs": {"corex_clock": str(bitcode)},
        "compiled_hash": str(compiled.hash),
        "dependency_verified": dependency_verified,
        "linked_llir_checks": checks,
        "artifact": artifact,
    }


def _artifact_hashes(
    compiled: Any,
    *,
    linked_llir: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {"llir": dict(linked_llir)}
    asm = getattr(compiled, "asm", {})
    for key in ("ttgir", "cubin"):
        payload = asm.get(key)
        if payload is None:
            records[key] = {"status": "unavailable"}
            continue
        data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        records[key] = {
            "status": "observed",
            "byte_count": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return records


def _resource_record(compiled: Any) -> dict[str, int]:
    return {
        "n_regs": int(compiled.n_regs),
        "n_spills": int(compiled.n_spills),
        "shared_bytes": int(compiled.metadata.shared),
    }


def _environment() -> dict[str, Any]:
    target = driver.active.get_current_target()
    properties = driver.active.utils.get_device_properties(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "corex": os.environ.get("COREX_VERSION", "unknown"),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "target": str(target),
        "warp_size": int(properties.get("warpSize", WARP_SIZE)),
    }


def _launch(
    kernel: Any,
    seed: Any,
    profile: Any,
    output: Any,
    *,
    chain_iters: int,
    clock_bitcode: Path,
) -> None:
    kernel[(1,)](
        seed,
        profile,
        output,
        GENERATION,
        chain_iters=chain_iters,
        num_warps=NUM_WARPS,
        extern_libs={"corex_clock": str(clock_bitcode)},
    )


def _run_specialization(
    kernel: Any,
    seed: Any,
    profile: Any,
    output: Any,
    *,
    chain_iters: int,
    samples: int,
    clock_bitcode: Path,
) -> tuple[Any, list[int], bool]:
    compiled = kernel.warmup(
        seed,
        profile,
        output,
        GENERATION,
        chain_iters=chain_iters,
        grid=(1,),
        num_warps=NUM_WARPS,
        extern_libs={"corex_clock": str(clock_bitcode)},
    )

    _launch(
        kernel,
        seed,
        profile,
        output,
        chain_iters=chain_iters,
        clock_bitcode=clock_bitcode,
    )
    torch.cuda.synchronize()
    expected = torch.tensor(
        [_expected_output(chain_iters)], dtype=torch.int64, device="cuda"
    )
    if not torch.equal(output, expected):
        raise ExternalClockProbeError(
            f"deterministic output mismatch for CHAIN_ITERS={chain_iters}"
        )

    for _ in range(WARMUP_RUNS):
        _launch(
            kernel,
            seed,
            profile,
            output,
            chain_iters=chain_iters,
            clock_bitcode=clock_bitcode,
        )
    torch.cuda.synchronize()

    deltas: list[int] = []
    writeback_ok = True
    for _ in range(samples):
        profile.zero_()
        _launch(
            kernel,
            seed,
            profile,
            output,
            chain_iters=chain_iters,
            clock_bitcode=clock_bitcode,
        )
        torch.cuda.synchronize()
        row = decode_program_slot(profile.cpu().tolist(), generation=GENERATION)
        if row["status"] != "observed":
            writeback_ok = False
            break
        deltas.append(int(row["raw_cycle_delta"]))
    return compiled, deltas, writeback_ok


def _control_region(
    *,
    label: str,
    chain_iters: int,
    representative_delta: int,
    summary: Mapping[str, int | float],
) -> dict[str, Any]:
    return {
        "pid": 0,
        "chain_variant": label,
        "chain_iters": chain_iters,
        "status": "observed",
        "cause": "none",
        "measurement_semantics": "issue-window",
        "start_boundary": "chain-start",
        "end_boundary": "token-dependent-chain-end",
        "raw_cycle_delta": representative_delta,
        "raw_cycle_median": summary["median"],
        "raw_cycle_minimum": summary["minimum"],
        "raw_cycle_p10": summary["p10"],
        "raw_cycle_p90": summary["p90"],
        "raw_cycle_maximum": summary["maximum"],
        "coefficient_of_variation": summary["coefficient_of_variation"],
        "sample_count": summary["count"],
    }


def record_static_dependency_audit(
    result: dict[str, Any],
    static_audit: Mapping[str, Any],
) -> bool:
    """Record the audit and return whether runtime tensor allocation may proceed."""
    result["static_dependency_audit"] = dict(static_audit)
    if bool(static_audit.get("dependency_verified")):
        return True
    result["experiment_status"] = "inconclusive"
    result["qualification_status"] = "inconclusive"
    result["status_causes"] = ["end-dependency-optimized-away"]
    result["error"] = (
        "compile-only linked LLIR did not preserve a chain-derived "
        "completion dependency before the end clock"
    )
    validate_document(result)
    return False


def run_probe(
    *,
    clock_bitcode: Path,
    clock_metadata: Path,
    artifact_dir: Path,
    samples: int = DEFAULT_SAMPLES,
    source_guard: Callable[[Path], dict[str, str]] = verify_accepted_sources,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    commit = current_commit(REPO_ROOT)

    # This statement must stay before device-library initialization.
    accepted_sources = source_guard(REPO_ROOT)
    helper = load_clock_helper(clock_bitcode, clock_metadata)
    result = build_result_skeleton(
        commit=commit,
        accepted_sources=accepted_sources,
        helper=helper,
    )

    _initialize_device_globals()
    kernel = _create_external_clock_kernel()
    bitcode = Path(helper["bitcode_absolute_path"])
    try:
        static_audit = compile_static_dependency_audit(
            kernel=kernel,
            clock_bitcode=bitcode,
            artifact_dir=artifact_dir,
        )
    except Exception as exc:
        result["experiment_status"] = "inconclusive"
        result["qualification_status"] = "inconclusive"
        result["status_causes"] = ["static-dependency-audit-failed"]
        result["static_dependency_audit"] = {
            "status": "failed",
            "mode": "compile-only-prelaunch",
            "chain_iters": STATIC_AUDIT_CHAIN_ITERS,
            "signature": dict(STATIC_AUDIT_SIGNATURE),
            "target": "GPUTarget(backend='cuda', arch=71, warp_size=64)",
            "error": compact_exception_text(exc),
        }
        validate_document(result)
        return result

    result["environment"].update(
        {
            "corex": os.environ.get("COREX_VERSION", "unknown"),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "target": static_audit["target"],
        }
    )
    if not record_static_dependency_audit(result, static_audit):
        return result

    result["environment"].update(_environment())
    seed = torch.tensor([RUNTIME_SEED], dtype=torch.int64, device="cuda")
    profile = torch.zeros(PROFILE_WORDS, dtype=torch.int64, device="cuda")
    output = torch.empty(OUTPUT_WORDS, dtype=torch.int64, device="cuda")

    control_data: dict[str, dict[str, Any]] = {}
    try:
        for chain_iters in CONTROL_CHAIN_ITERS:
            label = CONTROL_LABELS[chain_iters]
            compiled, deltas, writeback_ok = _run_specialization(
                kernel,
                seed,
                profile,
                output,
                chain_iters=chain_iters,
                samples=samples,
                clock_bitcode=bitcode,
            )
            if not deltas:
                raise ExternalClockProbeError(
                    f"no profile samples for CHAIN_ITERS={chain_iters}"
                )
            llir = compiled.asm.get("llir", "")
            if not isinstance(llir, str):
                llir = bytes(llir).decode("utf-8", "replace")
            llir_checks = inspect_linked_llir(
                llir,
                minimum_chain_steps=chain_iters,
            )
            linked_llir = persist_linked_llir(artifact_dir, label, llir)
            summary = summarize_cycles(deltas)
            resources = _resource_record(compiled)
            control_data[label] = {
                "chain_iters": chain_iters,
                "summary": summary,
                "deltas": deltas,
                "writeback_ok": writeback_ok,
                "compiled": compiled,
                "llir_checks": llir_checks,
                "linked_llir": linked_llir,
                "resources": resources,
            }
            result["regions"].append(
                _control_region(
                    label=label,
                    chain_iters=chain_iters,
                    representative_delta=deltas[0],
                    summary=summary,
                )
            )
    except Exception as exc:
        error = compact_exception_text(exc)
        result["experiment_status"] = "inconclusive"
        result["qualification_status"] = "inconclusive"
        result["status_causes"] = [classify_runtime_failure(error)]
        result["error"] = error
        validate_document(result)
        return result

    medians = {
        label: control_data[label]["summary"]["median"]
        for label in ("empty", "short", "long")
    }
    short_long_sensitive = medians["long"] > medians["short"] > medians["empty"]
    all_deltas = [
        delta
        for label in ("empty", "short", "long")
        for delta in control_data[label]["deltas"]
    ]
    linked = all(
        control_data[label]["llir_checks"]["linked"]
        for label in ("empty", "short", "long")
    )
    intrinsic_count_ok = all(
        control_data[label]["llir_checks"]["intrinsic_count_ok"]
        for label in ("empty", "short", "long")
    )
    dependency_verified = all(
        control_data[label]["llir_checks"]["chain_dependency_verified"]
        and control_data[label]["llir_checks"]["start_before_end"]
        for label in ("empty", "short", "long")
    )
    writeback_ok = all(
        control_data[label]["writeback_ok"]
        and len(control_data[label]["deltas"]) == samples
        for label in ("empty", "short", "long")
    )
    positive_deltas = bool(all_deltas) and all(delta > 0 for delta in all_deltas)
    no_spills = all(
        control_data[label]["resources"]["n_spills"] == 0
        for label in ("empty", "short", "long")
    )

    status, causes = classify_external_clock(
        linked=linked,
        intrinsic_count_ok=intrinsic_count_ok,
        writeback_ok=writeback_ok,
        positive_deltas=positive_deltas,
        short_long_sensitive=short_long_sensitive,
        dependency_verified=dependency_verified,
        no_spills=no_spills,
    )
    result["qualification_status"] = status
    result["experiment_status"] = status
    result["status_causes"] = causes
    result["sensitivity"] = {
        "median_issue_window": medians,
        "long_gt_short_gt_empty": short_long_sensitive,
        "all_deltas_positive": positive_deltas,
        "sample_count_per_control": samples,
    }
    result["compiler"] = {}
    for label in ("empty", "short", "long"):
        record = control_data[label]
        compiled = record["compiled"]
        result["compiler"][label] = {
            "compiled_hash": compiled.hash,
            "resources": record["resources"],
            "linked_llir_checks": record["llir_checks"],
            "artifacts": _artifact_hashes(
                compiled,
                linked_llir=record["linked_llir"],
            ),
        }
    validate_document(result)
    return result


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clock-bitcode",
        type=Path,
        default=DEFAULT_HELPER_DIR / "corex-clock.bc",
    )
    parser.add_argument(
        "--clock-metadata",
        type=Path,
        default=DEFAULT_HELPER_DIR / "clock-helper.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="linked LLIR directory; defaults to <output-dir>/linked-llir",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _empty_helper() -> dict[str, str]:
    return {
        "target": "unknown",
        "target_triple": "unknown",
        "source_sha256": "unknown",
        "bitcode_sha256": "unknown",
        "metadata_sha256": "unknown",
    }


def _source_guard_failure(commit: str, error: Exception) -> dict[str, Any]:
    payload = build_result_skeleton(
        commit=commit,
        accepted_sources={},
        helper=_empty_helper(),
    )
    payload["experiment_status"] = "invalid"
    payload["qualification_status"] = "invalid"
    payload["status_causes"] = ["accepted-source-mismatch"]
    payload["error"] = str(error)
    validate_document(payload)
    return payload


def _helper_failure_result(
    *,
    commit: str,
    accepted_sources: dict[str, str],
    error: Exception,
) -> dict[str, Any]:
    payload = build_result_skeleton(
        commit=commit,
        accepted_sources=accepted_sources,
        helper=_empty_helper(),
    )
    payload["status_causes"] = ["clock-helper-invalid"]
    payload["error"] = compact_exception_text(error)
    validate_document(payload)
    return payload


def _artifact_dir(args: argparse.Namespace) -> Path:
    return (
        Path(args.artifact_dir)
        if args.artifact_dir is not None
        else Path(args.output).parent / "linked-llir"
    )


def _worker_main(args: argparse.Namespace) -> int:
    commit = current_commit(REPO_ROOT)
    try:
        payload = run_probe(
            clock_bitcode=args.clock_bitcode,
            clock_metadata=args.clock_metadata,
            artifact_dir=_artifact_dir(args),
            samples=args.samples,
        )
    except SourceGuardError as exc:
        payload = _source_guard_failure(commit, exc)
    except Exception as exc:
        try:
            accepted_sources = verify_accepted_sources(REPO_ROOT)
        except SourceGuardError as guard_error:
            payload = _source_guard_failure(commit, guard_error)
        else:
            payload = _helper_failure_result(
                commit=commit,
                accepted_sources=accepted_sources,
                error=exc,
            )
    write_result(args.output, payload)
    return 0 if payload["experiment_status"] != "invalid" else 2


def read_worker_payload(
    worker_output: Path,
    completed: Any,
) -> dict[str, Any] | None:
    """Retain a written worker result even when teardown returns nonzero."""
    path = Path(worker_output)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(completed.returncode) != 0:
        payload["worker_returncode"] = int(completed.returncode)
    return payload


def _parent_main(args: argparse.Namespace) -> int:
    commit = current_commit(REPO_ROOT)
    try:
        accepted_sources = verify_accepted_sources(REPO_ROOT)
    except SourceGuardError as exc:
        payload = _source_guard_failure(commit, exc)
        write_result(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    try:
        helper = load_clock_helper(args.clock_bitcode, args.clock_metadata)
    except Exception as exc:
        payload = _helper_failure_result(
            commit=commit,
            accepted_sources=accepted_sources,
            error=exc,
        )
        write_result(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    artifact_dir = _artifact_dir(args).resolve()
    with tempfile.TemporaryDirectory(prefix="bi150-external-clock-") as directory:
        worker_output = Path(directory) / "worker-result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--clock-bitcode",
            str(Path(helper["bitcode_absolute_path"])),
            "--clock-metadata",
            str(Path(args.clock_metadata).resolve()),
            "--output",
            str(worker_output),
            "--artifact-dir",
            str(artifact_dir),
            "--samples",
            str(args.samples),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        payload = read_worker_payload(worker_output, completed)
        if payload is None:
            payload = build_result_skeleton(
                commit=commit,
                accepted_sources=accepted_sources,
                helper=helper,
            )
            payload["status_causes"] = ["external-clock-worker-failed"]
            payload["worker_returncode"] = completed.returncode
            payload["error"] = (
                "external-clock worker terminated during compilation or runtime: "
                + (completed.stderr or completed.stdout or "unknown worker failure")[-2000:]
            )
            validate_document(payload)

    write_result(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["experiment_status"] != "invalid" else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return _worker_main(args) if args.worker else _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

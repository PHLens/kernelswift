#!/usr/bin/env python3
"""Minimal BI150/CoreX Stage-0 observability probe.

This module intentionally imports no device libraries at module import time.
Accepted source hashes are verified before Torch or Triton are imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "lib"))

from result_contract import validate_document
from source_guard import SourceGuardError, verify_accepted_sources

N_ELEMENTS = 4096
BLOCK_SIZE = 256
NUM_WARPS = 1
WARMUP_RUNS = 20
DEFAULT_SAMPLES = 50


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_event_samples_us(samples: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in samples]
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("CUDA Event samples must be positive and finite")
    mean = statistics.fmean(values)
    return {
        "sample_count": len(values),
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.10),
        "p90_us": percentile(values, 0.90),
        "coefficient_of_variation": statistics.pstdev(values) / mean,
    }


def materialize_compiled_artifacts(compiled: Any, output_dir: Path) -> dict[str, dict[str, Any]]:
    """Write TTGIR, LLIR, and cubin from a compiled-like object."""
    artifact_dir = Path(output_dir) / "compiler"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    asm = getattr(compiled, "asm", {})
    result: dict[str, dict[str, Any]] = {}
    for key in ("ttgir", "llir", "cubin"):
        payload = asm.get(key)
        if payload is None:
            result[key] = {"status": "unavailable", "cause": f"{key}-missing"}
            continue
        if isinstance(payload, str):
            data = payload.encode("utf-8")
            suffix = key
        elif isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
            suffix = key
        else:
            result[key] = {"status": "unavailable", "cause": f"{key}-unsupported-type"}
            continue
        path = artifact_dir / f"stock-vector-add.{suffix}"
        path.write_bytes(data)
        result[key] = {
            "status": "observed",
            "relative_path": path.relative_to(output_dir).as_posix(),
            "byte_count": len(data),
            "sha256": sha256_bytes(data),
        }
    return result


def has_nonempty_function_body(text: str) -> bool:
    """Conservatively recognize an LLVM- or ixobjdump-style function body."""
    function_label = re.search(r"(?m)^\s*[0-9a-fA-F]+\s+<[^>]+>:\s*$", text)
    ix_function_label = re.search(r"(?im)^\s*function\s*(?::|=)\s*\S+", text)
    llvm_instruction = re.search(
        r"(?m)^\s*[0-9a-fA-F]+:\s+(?:[0-9a-fA-F]{2,}\s+)+\S+", text
    )
    ix_instruction = re.search(r"(?m)^\s*/\*[0-9a-fA-F]+\*/\s+\S+", text)
    return (function_label is not None and llvm_instruction is not None) or (
        ix_function_label is not None and ix_instruction is not None
    )


def disassembly_succeeded(returncode: int, stdout: str) -> bool:
    return returncode == 0 and has_nonempty_function_body(stdout)


def run_disassembler(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout_byte_count": len(stdout.encode("utf-8")),
            "stderr_tail": stderr[-1000:],
            "function_body_present": has_nonempty_function_body(stdout),
            "valid": disassembly_succeeded(completed.returncode, stdout),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout_byte_count": 0,
            "stderr_tail": str(exc),
            "function_body_present": False,
            "valid": False,
        }


def classify_preflight(checks: Mapping[str, bool]) -> tuple[str, list[str]]:
    """Classify only the capabilities needed by this Stage-0 probe."""
    if not checks.get("source_guard", False):
        return "invalid", ["accepted-source-mismatch"]

    cause_by_check = {
        "environment": "environment-unavailable",
        "compile": "stock-kernel-compile-failed",
        "correctness": "stock-kernel-correctness-failed",
        "resources": "resource-evidence-unavailable",
        "artifacts": "compiler-artifacts-unavailable",
        "event_timing": "kernel-event-timing-unavailable",
        "disassembly": "final-isa-unavailable",
    }
    causes = [cause for key, cause in cause_by_check.items() if not checks.get(key, False)]
    return ("valid", []) if not causes else ("inconclusive", causes)


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


def _invalid_source_result(commit: str, message: str) -> dict[str, Any]:
    return {
        "document_type": "preflight-result",
        "environment": {"route_c_commit": commit},
        "accepted_sources": {},
        "status": "invalid",
        "causes": ["accepted-source-mismatch"],
        "error": message,
    }


def _initialize_device_globals(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> None:
    """Lazily publish device libraries for Triton JIT function globals."""
    global torch, triton, tl, driver
    torch = import_module("torch")
    triton = import_module("triton")
    tl = import_module("triton.language")
    driver = import_module("triton.runtime").driver


def _create_stock_vector_add() -> Any:
    """Create the JIT kernel after its module globals have been initialized."""

    @triton.jit
    def stock_vector_add(x_ptr, y_ptr, out_ptr, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    return stock_vector_add


def run_preflight(
    output_dir: Path,
    *,
    samples: int = DEFAULT_SAMPLES,
    source_guard: Callable[[Path], dict[str, str]] = verify_accepted_sources,
) -> dict[str, Any]:
    """Run the Stage-0 probe and return its normalized result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = current_commit(REPO_ROOT)

    # This call must remain before the device-library imports below.
    accepted_sources = source_guard(REPO_ROOT)

    _initialize_device_globals()
    stock_vector_add = _create_stock_vector_add()

    checks = {
        "source_guard": True,
        "environment": False,
        "compile": False,
        "correctness": False,
        "resources": False,
        "artifacts": False,
        "event_timing": False,
        "disassembly": False,
    }
    result: dict[str, Any] = {
        "document_type": "preflight-result",
        "environment": {"route_c_commit": commit},
        "accepted_sources": accepted_sources,
        "stock_kernel": {
            "name": "stock_vector_add",
            "n_elements": N_ELEMENTS,
            "block_size": BLOCK_SIZE,
            "num_warps": NUM_WARPS,
            "launch_threads": NUM_WARPS * 64,
        },
        "causes": [],
        "status": "inconclusive",
    }

    try:
        target = driver.active.get_current_target()
        properties = driver.active.utils.get_device_properties(0)
        result["environment"].update(
            {
                "device": torch.cuda.get_device_name(0),
                "corex": os.environ.get("COREX_VERSION", "unknown"),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "target": str(target),
                "warp_size": int(properties.get("warpSize", 64)),
            }
        )
        checks["environment"] = True

        x = torch.arange(N_ELEMENTS, dtype=torch.float32, device="cuda")
        y = torch.arange(N_ELEMENTS, dtype=torch.float32, device="cuda") * 0.5
        out = torch.empty_like(x)
        grid = (triton.cdiv(N_ELEMENTS, BLOCK_SIZE),)
        compiled = stock_vector_add.warmup(
            x,
            y,
            out,
            n=N_ELEMENTS,
            block=BLOCK_SIZE,
            grid=grid,
            num_warps=NUM_WARPS,
        )
        checks["compile"] = True

        stock_vector_add[grid](
            x,
            y,
            out,
            n=N_ELEMENTS,
            block=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x + y, rtol=0, atol=0)
        checks["correctness"] = True
        result["correctness"] = {"status": "pass", "bitwise_equal": torch.equal(out, x + y)}

        resources = {
            "n_regs": int(compiled.n_regs),
            "n_spills": int(compiled.n_spills),
            "shared_bytes": int(compiled.metadata.shared),
            "n_threads_attribute": int(compiled.n_threads),
        }
        checks["resources"] = all(value >= 0 for value in resources.values())
        result["stock_kernel"].update(
            {
                "compiled_hash": compiled.hash,
                "asm_keys": sorted(compiled.asm.keys()),
                "resources": resources,
            }
        )

        artifacts = materialize_compiled_artifacts(compiled, output_dir)
        result["artifacts"] = artifacts
        checks["artifacts"] = all(
            artifacts[key].get("status") == "observed" and artifacts[key].get("byte_count", 0) > 0
            for key in ("ttgir", "llir", "cubin")
        )

        for _ in range(WARMUP_RUNS):
            stock_vector_add[grid](
                x,
                y,
                out,
                n=N_ELEMENTS,
                block=BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )
        torch.cuda.synchronize()

        event_samples_us: list[float] = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            stock_vector_add[grid](
                x,
                y,
                out,
                n=N_ELEMENTS,
                block=BLOCK_SIZE,
                num_warps=NUM_WARPS,
            )
            end.record()
            end.synchronize()
            event_samples_us.append(float(start.elapsed_time(end)) * 1000.0)
        result["cuda_event_timing"] = summarize_event_samples_us(event_samples_us)
        checks["event_timing"] = True

        cubin_path = output_dir / artifacts["cubin"]["relative_path"]
        commands = [
            ["/usr/local/corex-4.4.0/bin/llvm-objdump", "-d", str(cubin_path)],
            ["/usr/local/corex-4.4.0/bin/ixobjdump", "--sass", str(cubin_path)],
        ]
        attempts = [run_disassembler(command) for command in commands]
        result["disassembly_attempts"] = attempts
        checks["disassembly"] = any(attempt["valid"] for attempt in attempts)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    status, causes = classify_preflight(checks)
    result["status"] = status
    result["causes"] = causes
    return result


def write_result(path: Path, payload: dict[str, Any]) -> None:
    validate_document(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    commit = current_commit(REPO_ROOT)
    try:
        payload = run_preflight(output_dir, samples=args.samples)
    except SourceGuardError as exc:
        payload = _invalid_source_result(commit, str(exc))
    result_path = output_dir / "stage0-result.json"
    write_result(result_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"valid", "inconclusive"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

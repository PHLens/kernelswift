#!/usr/bin/env python3
"""Ascend 910B matrix.dot fp16 tile-coverage probe payload.

Compiles and executes ``tl.dot`` with fp16 inputs and fp32 accumulation across
power-of-two, multiple-of-16, and non-multiple-of-16 tile shapes on the matched
Ascend910B runtime, numerically checks each result against a torch reference,
and writes a normalized result payload to ``--result-json``.

The recorded prior evidence for this target covers only ``(16,16)@(16,16)`` in
fp32, while mm_encoder_attention, flexattention, and fused_moe all need an fp16
dot path at head_size 64 and hidden 128 with a sequence length of 83 that must
be padded or tiled. The primary capability is therefore fp16 tile coverage;
fp32 and bf16 behaviour is recorded in the detail text as advisory context only
and does not qualify a separate capability.
"""

import argparse
import json
import sys

try:
    import torch
    import torch_npu  # noqa: F401  required before any NPU allocation
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    torch = None
    torch_npu = None
    triton = None
    tl = None


PRIMARY_DTYPE = "fp16"
ADVISORY_DTYPES = ["fp32", "bf16"]
# (M, N, K) tiles grouped by the arithmetic constraint they test.
POWER_OF_TWO = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 64)]
MULTIPLE_OF_16 = [(48, 48, 16), (80, 80, 32), (96, 96, 64), (112, 64, 32)]
NON_MULTIPLE_OF_16 = [(83, 64, 64), (64, 83, 64), (64, 64, 83)]
TOLERANCE = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 5e-2}


def _torch_dtype(name: str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _run_case(dot_kernel, M, N, K, dtype, device):
    torch_dtype = _torch_dtype(dtype)
    try:
        A = torch.randn(M, K, dtype=torch_dtype, device=device)
        B = torch.randn(K, N, dtype=torch_dtype, device=device)
        C = torch.zeros(M, N, dtype=torch.float32, device=device)
        dot_kernel[(1,)](A, B, C, M=M, N=N, K=K)
        torch.npu.synchronize()
        ref = A.float() @ B.float()
        err = (C - ref).abs().max().item()
        return {
            "shape": [M, N, K],
            "dtype": dtype,
            "compiled": True,
            "numerically_checked": err < TOLERANCE[dtype],
            "max_abs_diff": err,
        }
    except Exception as error:  # noqa: BLE001
        return {
            "shape": [M, N, K],
            "dtype": dtype,
            "compiled": False,
            "numerically_checked": False,
            "error": str(error)[:200],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--runtime-snapshot", required=True)
    args = parser.parse_args()

    probe_id = "triton-ascend-dot-tiles-001"
    profile_id = "triton_ascend"
    capability_id = "matrix.dot.fp16-fp16-fp32.tile-coverage"

    try:
        with open(args.runtime_snapshot, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        if tl is None:
            raise RuntimeError("triton or torch_npu import failed; probe cannot run")

        device = snapshot.get("device") or "npu"

        @triton.jit
        def dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
            a = tl.load(A + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
            b = tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
            c = tl.dot(a, b)
            tl.store(C + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], c)

        groups = {
            "power-of-two": POWER_OF_TWO,
            "multiple-of-16": MULTIPLE_OF_16,
            "non-multiple-of-16": NON_MULTIPLE_OF_16,
        }

        primary_details = []
        for group, shapes in groups.items():
            for M, N, K in shapes:
                result = _run_case(dot_kernel, M, N, K, PRIMARY_DTYPE, device)
                result["group"] = group
                primary_details.append(result)

        advisory_details = []
        for dtype in ADVISORY_DTYPES:
            for M, N, K in POWER_OF_TWO:
                result = _run_case(dot_kernel, M, N, K, dtype, device)
                result["group"] = "power-of-two-advisory"
                advisory_details.append(result)

        passing = [
            (tuple(item["shape"]))
            for item in primary_details
            if item["numerically_checked"]
        ]
        advisory_passing = sorted(
            {
                (tuple(item["shape"]), item["dtype"])
                for item in advisory_details
                if item["numerically_checked"]
            }
        )
        compiled_any = any(item["compiled"] for item in primary_details)
        checked_any = any(item["numerically_checked"] for item in primary_details)
        level = "observed" if compiled_any else "unknown"
        detail = (
            "fp16 tl.dot compiled on %d/%d cases; %d numerically checked; passing tiles: %s. "
            "Advisory fp32/bf16 passing (shape, dtype) pairs: %s"
            % (
                sum(1 for item in primary_details if item["compiled"]),
                len(primary_details),
                len(passing),
                passing or "none",
                advisory_passing or "none",
            )
        )
        observations = [
            {
                "capability_id": capability_id,
                "level": level,
                "numerically_checked": checked_any,
                "detail": detail,
                "shape_details": primary_details,
                "advisory_shape_details": advisory_details,
            }
        ]
        payload = {
            "schema_version": 1,
            "probe_id": probe_id,
            "implementation_profile_id": profile_id,
            "target_id": args.target_id,
            "observed_scope": {
                "accumulator_dtype": "fp32",
                "lhs_dtype": "fp16",
                "rhs_dtype": "fp16",
                "tile_groups": sorted(groups),
            },
            "observations": observations,
        }
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 0 if checked_any else 1
    except Exception as error:  # noqa: BLE001
        payload = {
            "schema_version": 1,
            "probe_id": probe_id,
            "implementation_profile_id": profile_id,
            "target_id": args.target_id,
            "observed_scope": {
                "accumulator_dtype": "fp32",
                "lhs_dtype": "fp16",
                "rhs_dtype": "fp16",
            },
            "observations": [
                {
                    "capability_id": capability_id,
                    "level": "unknown",
                    "numerically_checked": False,
                    "detail": str(error)[:400],
                }
            ],
        }
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

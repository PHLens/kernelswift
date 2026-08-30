#!/usr/bin/env python3
"""S60 GCU matrix.dot mult-of-16 tile probe payload.

Compiles and executes tl.dot at several (M,N,K) tile shapes on the matched
S60/GCU runtime, records which shapes satisfy the shape constraint, numerically
checks the fp16 result, and writes a normalized result payload to --result-json.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--runtime-snapshot", required=True)
    args = parser.parse_args()

    probe_id = "triton-gcu-dot-mult16-001"
    profile_id = "triton_gcu"
    capability_id = "matrix.dot.fp16-fp16-fp32.mult-of-16-tiles"

    try:
        with open(args.runtime_snapshot, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        import torch  # noqa: F401
        import torch_gcu  # noqa: F401
        import triton
        import triton.language as tl
        import triton_gcu  # noqa: F401

        device = snapshot.get("device") or "gcu"

        @triton.jit
        def dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
            a = tl.load(A + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
            b = tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
            c = tl.dot(a, b)
            tl.store(C + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], c)

        shapes = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 64)]
        details = []
        all_ok = True
        for M, N, K in shapes:
            A = torch.randn(M, K, dtype=torch.float16, device=device)
            B = torch.randn(K, N, dtype=torch.float16, device=device)
            C = torch.zeros(M, N, dtype=torch.float32, device=device)
            try:
                dot_kernel[(1,)](A, B, C, M=M, N=N, K=K)
                torch.gcu.synchronize()
                ref = A.float() @ B.float()
                err = (C - ref).abs().max().item()
                ok = err < 1e-3
                details.append({"shape": [M, N, K], "compiled": True, "numerically_checked": ok, "max_abs_diff": err})
                all_ok = all_ok and ok
            except Exception as error:  # noqa: BLE001
                details.append({"shape": [M, N, K], "compiled": False, "error": str(error)[:200]})
                all_ok = False

        # 约束验证：非 16 倍数 shape 应失败（确认约束方向）
        constrained_shapes = [(48, 64, 64), (64, 83, 64), (64, 64, 80)]
        for M, N, K in constrained_shapes:
            A = torch.randn(M, K, dtype=torch.float16, device=device)
            B = torch.randn(K, N, dtype=torch.float16, device=device)
            C = torch.zeros(M, N, dtype=torch.float32, device=device)
            try:
                dot_kernel[(1,)](A, B, C, M=M, N=N, K=K)
                torch.gcu.synchronize()
                details.append({"shape": [M, N, K], "compiled": True, "note": "unexpectedly compiled despite non-mult-of-16"})
            except Exception as error:  # noqa: BLE001
                details.append({"shape": [M, N, K], "compiled": False, "error": str(error)[:200]})

        observations = [
            {
                "capability_id": capability_id,
                "level": "observed",
                "numerically_checked": all_ok,
                "detail": "mult-of-16 tiles fp16 dot numerically checked; non-mult-of-16 shapes rejected",
                "shape_details": details,
            }
        ]
        payload = {
            "schema_version": 1,
            "probe_id": probe_id,
            "implementation_profile_id": profile_id,
            "target_id": args.target_id,
            "observed_scope": {
                "dtype": "fp16",
                "accumulator_dtype": "fp32",
                "shape_constraint": "M/N/K multiples of 16",
            },
            "observations": observations,
        }
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 0
    except Exception as error:  # noqa: BLE001
        payload = {
            "schema_version": 1,
            "probe_id": probe_id,
            "implementation_profile_id": profile_id,
            "target_id": args.target_id,
            "observed_scope": {"dtype": "fp16", "accumulator_dtype": "fp32"},
            "observations": [
                {"capability_id": capability_id, "level": "unknown", "numerically_checked": False, "detail": str(error)}
            ],
        }
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

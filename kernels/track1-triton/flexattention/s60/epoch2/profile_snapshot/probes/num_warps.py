#!/usr/bin/env python3
"""S60 GCU num_warps probe payload.

Compiles and executes a simple elementwise kernel under num_warps 1/2/4/8 on the
matched S60/GCU runtime, numerically checks each, and writes a normalized result
payload to --result-json.
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

    probe_id = "triton-gcu-num-warps-001"
    profile_id = "triton_gcu"
    capability_id = "resource.num-warps"

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
        def ew_kernel(A, B, N: tl.constexpr):
            offsets = tl.arange(0, N)
            v = tl.load(A + offsets)
            tl.store(B + offsets, v * 2.0 + 1.0)

        N = 1024
        A = torch.randn(N, dtype=torch.float32, device=device)
        results = []
        all_ok = True
        for nw in [1, 2, 4, 8]:
            B = torch.zeros(N, dtype=torch.float32, device=device)
            try:
                ew_kernel[(1,)](A, B, N=N, num_warps=nw)
                torch.gcu.synchronize()
                ok = bool(torch.equal(B, A * 2.0 + 1.0))
                results.append({"num_warps": nw, "compiled": True, "numerically_checked": ok})
                all_ok = all_ok and ok
            except Exception as error:  # noqa: BLE001
                results.append({"num_warps": nw, "compiled": False, "error": str(error)[:200]})
                all_ok = False

        observations = [
            {
                "capability_id": capability_id,
                "level": "observed",
                "numerically_checked": all_ok,
                "detail": "num_warps 1/2/4/8 compiled and numerically checked",
                "warp_details": results,
            }
        ]
        payload = {
            "schema_version": 1,
            "probe_id": probe_id,
            "implementation_profile_id": profile_id,
            "target_id": args.target_id,
            "observed_scope": {"name": "num_warps", "values": [1, 2, 4, 8]},
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
            "observed_scope": {"name": "num_warps"},
            "observations": [
                {"capability_id": capability_id, "level": "unknown", "numerically_checked": False, "detail": str(error)}
            ],
        }
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

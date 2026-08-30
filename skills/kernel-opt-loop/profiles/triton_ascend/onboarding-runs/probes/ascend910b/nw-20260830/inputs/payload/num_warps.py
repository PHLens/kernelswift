#!/usr/bin/env python3
"""Ascend 910B resource.num-warps probe payload.

Compiles and executes one representative Triton kernel at num_warps 1, 2, 4, and
8 on the matched Ascend910B runtime, numerically checks each launch, and writes a
normalized result payload to ``--result-json``.

Larger warps are the remaining occupancy lever for the attention and GEMM
candidates once a dot path is available, so the legal value set must be
established rather than assumed from another backend.
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


WARPS = [1, 2, 4, 8]
BLOCK = 128


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--runtime-snapshot", required=True)
    args = parser.parse_args()

    probe_id = "triton-ascend-num-warps-001"
    profile_id = "triton_ascend"
    capability_id = "resource.num-warps"

    try:
        with open(args.runtime_snapshot, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        if tl is None:
            raise RuntimeError("triton or torch_npu import failed; probe cannot run")

        device = snapshot.get("device") or "npu"

        @triton.jit
        def scale_kernel(X, Y, BLOCK: tl.constexpr):
            offset = tl.arange(0, BLOCK)
            value = tl.load(X + offset)
            tl.store(Y + offset, value * 2.0)

        x = torch.arange(BLOCK, dtype=torch.float32, device=device)
        reference = x * 2.0

        details = []
        all_ok = True
        for warps in WARPS:
            try:
                y = torch.zeros(BLOCK, dtype=torch.float32, device=device)
                scale_kernel[(1,)](x, y, BLOCK=BLOCK, num_warps=warps)
                torch.npu.synchronize()
                ok = bool(torch.allclose(y, reference))
                details.append({"num_warps": warps, "compiled": True, "numerically_checked": ok})
                all_ok = all_ok and ok
            except Exception as error:  # noqa: BLE001
                details.append(
                    {
                        "num_warps": warps,
                        "compiled": False,
                        "numerically_checked": False,
                        "error": str(error)[:200],
                    }
                )
                all_ok = False

        compiled_any = any(item["compiled"] for item in details)
        level = "observed" if compiled_any else "unknown"
        passing = [item["num_warps"] for item in details if item["numerically_checked"]]
        observations = [
            {
                "capability_id": capability_id,
                "level": level,
                "numerically_checked": all_ok,
                "detail": "num_warps compiled and numerically checked for values %s at BLOCK=%d"
                % (passing or "none", BLOCK),
                "shape_details": details,
            }
        ]
        payload = {
            "schema_version": 1,
            "probe_id": probe_id,
            "implementation_profile_id": profile_id,
            "target_id": args.target_id,
            "observed_scope": {"dtype": "fp32", "block": BLOCK, "values": WARPS},
            "observations": observations,
        }
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 0 if all_ok else 1
    except Exception as error:  # noqa: BLE001
        payload = {
            "schema_version": 1,
            "probe_id": probe_id,
            "implementation_profile_id": profile_id,
            "target_id": args.target_id,
            "observed_scope": {"dtype": "fp32"},
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

#!/usr/bin/env python3
"""Ascend 910B resource.num-stages probe payload.

Compiles and executes a pipelined loop kernel at num_stages 1, 2, 3, and 4 on the
matched Ascend910B runtime, numerically checks each launch, and writes a
normalized result payload to ``--result-json``.

num_stages is recorded as Unknown for this target. Software pipelining is a
candidate lever for the attention and GEMM loops, so legality must be observed
before a design may depend on it.
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


STAGES = [1, 2, 3, 4]
BLOCK = 64
ITERATIONS = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--runtime-snapshot", required=True)
    args = parser.parse_args()

    probe_id = "triton-ascend-num-stages-001"
    profile_id = "triton_ascend"
    capability_id = "resource.num-stages"

    try:
        with open(args.runtime_snapshot, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        if tl is None:
            raise RuntimeError("triton or torch_npu import failed; probe cannot run")

        device = snapshot.get("device") or "npu"

        @triton.jit
        def accum_kernel(X, Y, BLOCK: tl.constexpr, ITERATIONS: tl.constexpr):
            offset = tl.arange(0, BLOCK)
            total = tl.zeros((BLOCK,), dtype=tl.float32)
            for step in tl.static_range(ITERATIONS):
                total += tl.load(X + offset) + step
            tl.store(Y + offset, total)

        x = torch.ones(BLOCK, dtype=torch.float32, device=device)
        reference = (x * ITERATIONS) + float(sum(range(ITERATIONS)))

        details = []
        all_ok = True
        for stages in STAGES:
            try:
                y = torch.zeros(BLOCK, dtype=torch.float32, device=device)
                accum_kernel[(1,)](
                    x, y, BLOCK=BLOCK, ITERATIONS=ITERATIONS, num_stages=stages
                )
                torch.npu.synchronize()
                ok = bool(torch.allclose(y, reference))
                details.append({"num_stages": stages, "compiled": True, "numerically_checked": ok})
                all_ok = all_ok and ok
            except Exception as error:  # noqa: BLE001
                details.append(
                    {
                        "num_stages": stages,
                        "compiled": False,
                        "numerically_checked": False,
                        "error": str(error)[:200],
                    }
                )
                all_ok = False

        compiled_any = any(item["compiled"] for item in details)
        level = "observed" if compiled_any else "unknown"
        passing = [item["num_stages"] for item in details if item["numerically_checked"]]
        observations = [
            {
                "capability_id": capability_id,
                "level": level,
                "numerically_checked": all_ok,
                "detail": "num_stages compiled and numerically checked for values %s at BLOCK=%d"
                % (passing or "none", BLOCK),
                "shape_details": details,
            }
        ]
        payload = {
            "schema_version": 1,
            "probe_id": probe_id,
            "implementation_profile_id": profile_id,
            "target_id": args.target_id,
            "observed_scope": {"dtype": "fp32", "block": BLOCK, "values": STAGES},
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

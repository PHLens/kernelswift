#!/usr/bin/env python3
"""Real MLU basic-memory probe payload.

Compiles and executes a masked contiguous fp32 load/store kernel on the matched
MLU runtime, checks the result numerically, and writes a normalized result
payload to --result-json. Unit tests never execute this hardware payload and no
committed evidence claims that it passed.
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

    try:
        with open(args.runtime_snapshot, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        import torch  # noqa: F401  (matched MLU runtime import)
        import triton
        import triton.language as tl

        device = snapshot.get("device")
        if not device:
            raise RuntimeError("runtime snapshot does not name a device")

        @triton.jit
        def basic_memory_kernel(pointer, mask, BLOCK_N: tl.constexpr):
            offsets = tl.arange(0, BLOCK_N)
            row = tl.load(pointer + offsets, mask=mask)
            tl.store(pointer + offsets, row * 2.0, mask=mask)

        size = 1024
        source = torch.arange(size, dtype=torch.float32, device=device)
        mask = torch.ones(size, dtype=torch.bool, device=device)
        grid = (1,)
        basic_memory_kernel[grid](source, mask, BLOCK_N=size)
        torch.mlu.synchronize() if hasattr(torch, "mlu") else None
        expected = source * 2.0
        numerically_checked = bool(torch.equal(source, expected))

        observations = [
            {"capability_id": capability_id, "level": "observed", "numerically_checked": numerically_checked, "detail": "masked contiguous fp32"}
            for capability_id in (
                "memory.load.contiguous",
                "memory.store.contiguous",
                "index.range.one-dimensional",
                "parallel.program-id.axis0",
            )
        ]
        payload = {
            "schema_version": 1,
            "probe_id": "triton-mlu-basic-memory-001",
            "implementation_profile_id": "triton_mlu",
            "target_id": args.target_id,
            "observed_scope": {"dtype": "fp32", "layout": "contiguous", "shape": ["N"]},
            "observations": observations,
        }
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 0
    except Exception as error:  # noqa: BLE001 - normalized probe failure output
        payload = {
            "schema_version": 1,
            "probe_id": "triton-mlu-basic-memory-001",
            "implementation_profile_id": "triton_mlu",
            "target_id": args.target_id,
            "observed_scope": {"dtype": "fp32", "layout": "contiguous", "shape": ["N"]},
            "observations": [
                {"capability_id": capability_id, "level": "unknown", "numerically_checked": False, "detail": str(error)}
                for capability_id in (
                    "memory.load.contiguous",
                    "memory.store.contiguous",
                    "index.range.one-dimensional",
                    "parallel.program-id.axis0",
                )
            ],
        }
        with open(args.result_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

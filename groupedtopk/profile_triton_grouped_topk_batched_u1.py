from __future__ import annotations

from pathlib import Path

import torch

from triton_grouped_topk_batched_u1 import grouped_topk_triton_batched_u1_out
from triton_grouped_topk_batched_sweep import (
    grouped_topk_batched2_out,
    grouped_topk_batched4_out,
)
from triton_grouped_topk_hierarchical import grouped_topk_triton_compact128_out


def main() -> None:
    torch.manual_seed(0)
    logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
    weights = torch.empty((83, 8), device="mlu", dtype=torch.float32)
    ids = torch.empty((83, 8), device="mlu", dtype=torch.int32)
    variants = (
        (
            "compact128",
            lambda: grouped_topk_triton_compact128_out(logits, weights, ids),
        ),
        (
            "batched_u1_w1",
            lambda: grouped_topk_triton_batched_u1_out(
                logits, weights, ids, num_warps=1
            ),
        ),
        (
            "batched_u1_w4",
            lambda: grouped_topk_triton_batched_u1_out(
                logits, weights, ids, num_warps=4
            ),
        ),
        ("batched2", lambda: grouped_topk_batched2_out(logits, weights, ids)),
        ("batched4", lambda: grouped_topk_batched4_out(logits, weights, ids)),
    )

    for _, fn in variants:
        for _ in range(20):
            fn()
    torch.mlu.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]
    with torch.no_grad(), torch.profiler.profile(activities=activities) as prof:
        for label, fn in variants:
            with torch.profiler.record_function(label):
                for _ in range(100):
                    fn()
        torch.mlu.synchronize()

    output = Path(
        "log/triton_grouped_topk_batched_u1_vs_compact128_T83_100iter.pt.trace.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(output))
    print(output)


if __name__ == "__main__":
    main()

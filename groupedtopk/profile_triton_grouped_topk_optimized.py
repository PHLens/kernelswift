from __future__ import annotations

import argparse
from pathlib import Path

import torch

from triton_grouped_topk import grouped_topk_triton_out
from triton_grouped_topk_optimized import grouped_topk_triton_optimized_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=83)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "log/triton_grouped_topk_group_rank_fixed_T83_"
            "preallocated_50iter.pt.trace.json"
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    logits = torch.randn((args.tokens, 256), device="mlu", dtype=torch.float32)
    weights = torch.empty((args.tokens, 8), device="mlu", dtype=torch.float32)
    ids = torch.empty((args.tokens, 8), device="mlu", dtype=torch.int32)
    candidates = (
        ("baseline", lambda: grouped_topk_triton_out(logits, weights, ids)),
        (
            "optimized",
            lambda: grouped_topk_triton_optimized_out(logits, weights, ids),
        ),
    )

    for _, fn in candidates:
        for _ in range(20):
            fn()
    torch.mlu.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]
    with torch.no_grad(), torch.profiler.profile(activities=activities) as prof:
        for label, fn in candidates:
            with torch.profiler.record_function(label):
                for _ in range(args.iterations):
                    fn()
        torch.mlu.synchronize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.output))
    print(args.output)


if __name__ == "__main__":
    main()

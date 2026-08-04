from __future__ import annotations

import argparse
from pathlib import Path

import torch

from triton_grouped_topk_hierarchical import grouped_topk_triton_compact128_out
from triton_grouped_topk_compact_variants import (
    grouped_topk_triton_compact128_shared_out,
    grouped_topk_triton_prefix_out,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("log/triton_grouped_topk_compact_variants_T83_100iter.pt.trace.json"),
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
    weights = torch.empty((83, 8), device="mlu", dtype=torch.float32)
    ids = torch.empty((83, 8), device="mlu", dtype=torch.int32)
    variants = (
        ("compact128", grouped_topk_triton_compact128_out),
        ("prefix", grouped_topk_triton_prefix_out),
        ("compact128_shared", grouped_topk_triton_compact128_shared_out),
    )
    for _, fn in variants:
        for _ in range(30):
            fn(logits, weights, ids)
    torch.mlu.synchronize()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]
    with torch.no_grad(), torch.profiler.profile(activities=activities) as prof:
        for label, fn in variants:
            with torch.profiler.record_function(label):
                for _ in range(args.iterations):
                    fn(logits, weights, ids)
        torch.mlu.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.output))
    print(args.output)


if __name__ == "__main__":
    main()


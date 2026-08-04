from __future__ import annotations

import argparse
import statistics
import time

import torch

from triton_grouped_topk_batched_u1 import grouped_topk_triton_batched_u1_out
from triton_grouped_topk_hierarchical import grouped_topk_triton_compact128_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(0)
    logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
    weights = torch.empty((83, 8), device="mlu", dtype=torch.float32)
    ids = torch.empty((83, 8), device="mlu", dtype=torch.int32)
    variants = (
        ("compact128", lambda: grouped_topk_triton_compact128_out(logits, weights, ids)),
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
    )

    for _, fn in variants:
        for _ in range(args.warmup):
            fn()
    torch.mlu.synchronize()

    samples = {label: [] for label, _ in variants}
    for repeat in range(args.repeats):
        order = variants if repeat % 2 == 0 else variants[::-1]
        for label, fn in order:
            torch.mlu.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                fn()
            torch.mlu.synchronize()
            samples[label].append(
                (time.perf_counter() - start) * 1e6 / args.iterations
            )

    for label, values in samples.items():
        print(
            f"{label:24s} median={statistics.median(values):8.3f} us "
            f"samples={[round(value, 3) for value in values]}"
        )


if __name__ == "__main__":
    main()

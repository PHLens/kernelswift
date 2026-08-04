from __future__ import annotations

import argparse
import statistics
import time

import torch

from triton_grouped_topk_u1 import (
    row_max_batched_u1_out,
    row_max_single_out,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(0)
    logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
    expected = logits.max(dim=1).values
    outputs = {
        "single_w1": torch.empty((83,), device="mlu", dtype=torch.float32),
        "batch_w1": torch.empty((83,), device="mlu", dtype=torch.float32),
        "batch_w4": torch.empty((83,), device="mlu", dtype=torch.float32),
        "batch_w4_shared": torch.empty((83,), device="mlu", dtype=torch.float32),
    }
    variants = (
        ("single_w1", lambda: row_max_single_out(logits, outputs["single_w1"])),
        (
            "batch_w1",
            lambda: row_max_batched_u1_out(
                logits, outputs["batch_w1"], num_warps=1
            ),
        ),
        (
            "batch_w4",
            lambda: row_max_batched_u1_out(
                logits, outputs["batch_w4"], num_warps=4
            ),
        ),
        (
            "batch_w4_shared",
            lambda: row_max_batched_u1_out(
                logits,
                outputs["batch_w4_shared"],
                num_warps=4,
                force_use_shared_memory=True,
            ),
        ),
    )

    for label, fn in variants:
        fn()
        torch.mlu.synchronize()
        if not torch.equal(outputs[label].cpu(), expected.cpu()):
            raise AssertionError(f"{label} correctness check failed")

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

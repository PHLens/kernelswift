from __future__ import annotations

import argparse
import statistics
import time

import torch

from base import Model
from triton_grouped_topk import grouped_topk_triton_out
from triton_grouped_topk_optimized import grouped_topk_triton_optimized_out


def benchmark(fn, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.mlu.synchronize()
    samples = []
    for _ in range(repeats):
        torch.mlu.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.mlu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=83)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    logits = torch.randn((args.tokens, 256), device="mlu", dtype=torch.float32)
    hidden = torch.empty((args.tokens, 1), device="mlu", dtype=torch.float16)
    weights = torch.empty((args.tokens, 8), device="mlu", dtype=torch.float32)
    ids = torch.empty((args.tokens, 8), device="mlu", dtype=torch.int32)
    expected_weights, expected_ids = Model(8, True, 8, 4)(hidden, logits)

    variants = (
        ("baseline", lambda: grouped_topk_triton_out(logits, weights, ids)),
        (
            "group_rank",
            lambda: grouped_topk_triton_optimized_out(logits, weights, ids),
        ),
    )

    with torch.no_grad():
        for label, fn in variants:
            fn()
            torch.mlu.synchronize()
            max_diff = (weights.cpu() - expected_weights.cpu()).abs().max().item()
            ids_equal = torch.equal(ids.cpu(), expected_ids.cpu())
            if not ids_equal or max_diff > 1e-6:
                raise AssertionError(
                    f"{label}: ids_equal={ids_equal}, max_diff={max_diff}"
                )
            samples = benchmark(fn, args.warmup, args.iterations, args.repeats)
            print(
                f"{label:18s} median={statistics.median(samples):8.3f} us "
                f"samples={[round(value, 3) for value in samples]}"
            )


if __name__ == "__main__":
    main()

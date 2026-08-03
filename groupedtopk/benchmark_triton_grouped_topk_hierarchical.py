from __future__ import annotations

import argparse
import statistics
import time

import torch

from base import Model
from triton_grouped_topk_hierarchical import (
    grouped_topk_triton_compact128_out,
    grouped_topk_triton_hierarchical_out,
    grouped_topk_triton_hierarchical_sort_out,
)
from triton_grouped_topk_optimized import grouped_topk_triton_optimized_out


def _check_result(
    label: str,
    actual_weights: torch.Tensor,
    actual_ids: torch.Tensor,
    expected_weights: torch.Tensor,
    expected_ids: torch.Tensor,
) -> None:
    actual_weights_cpu = actual_weights.cpu()
    actual_ids_cpu = actual_ids.cpu()
    expected_weights_cpu = expected_weights.cpu()
    expected_ids_cpu = expected_ids.cpu()
    max_diff = (actual_weights_cpu - expected_weights_cpu).abs().max().item()
    ids_equal = torch.equal(actual_ids_cpu, expected_ids_cpu)
    weights_close = torch.allclose(
        actual_weights_cpu, expected_weights_cpu, rtol=2e-5, atol=1e-6
    )
    print(
        f"{label:24s} ids_equal={ids_equal} "
        f"weights_close={weights_close} max_diff={max_diff:.3e}"
    )
    if not ids_equal or not weights_close:
        raise AssertionError(f"{label} correctness check failed")


def check_correctness() -> None:
    reference = Model(8, True, 8, 4).eval()
    candidates = (
        ("winner_tree", grouped_topk_triton_hierarchical_out),
        ("sort_32_64", grouped_topk_triton_hierarchical_sort_out),
        ("compact128", grouped_topk_triton_compact128_out),
    )

    with torch.no_grad():
        for seed in (0, 1, 17):
            torch.manual_seed(seed)
            logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
            hidden = torch.empty((83, 1), device="mlu", dtype=torch.float16)
            expected_weights, expected_ids = reference(hidden, logits)
            for label, fn in candidates:
                weights = torch.empty(
                    (83, 8), device="mlu", dtype=torch.float32
                )
                ids = torch.empty((83, 8), device="mlu", dtype=torch.int32)
                fn(logits, weights, ids)
                torch.mlu.synchronize()
                _check_result(
                    f"{label}/seed={seed}",
                    weights,
                    ids,
                    expected_weights,
                    expected_ids,
                )

        base_logits = torch.arange(256, device="mlu", dtype=torch.float32)
        logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
        logits[0] = base_logits
        logits[1] = -base_logits
        logits[2] = base_logits.remainder(17)
        logits[3] = 0
        expected_weights = torch.empty(
            (83, 8), device="mlu", dtype=torch.float32
        )
        expected_ids = torch.empty((83, 8), device="mlu", dtype=torch.int32)
        grouped_topk_triton_optimized_out(
            logits, expected_weights, expected_ids
        )
        for label, fn in candidates:
            weights = torch.empty((83, 8), device="mlu", dtype=torch.float32)
            ids = torch.empty((83, 8), device="mlu", dtype=torch.int32)
            fn(logits, weights, ids)
            torch.mlu.synchronize()
            _check_result(
                f"{label}/edges",
                weights,
                ids,
                expected_weights,
                expected_ids,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    check_correctness()
    torch.manual_seed(0)
    logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
    weights = torch.empty((83, 8), device="mlu", dtype=torch.float32)
    ids = torch.empty((83, 8), device="mlu", dtype=torch.int32)
    variants = (
        (
            "current",
            lambda: grouped_topk_triton_optimized_out(logits, weights, ids),
        ),
        (
            "winner_tree",
            lambda: grouped_topk_triton_hierarchical_out(logits, weights, ids),
        ),
        (
            "sort_32_64",
            lambda: grouped_topk_triton_hierarchical_sort_out(
                logits, weights, ids
            ),
        ),
        (
            "compact128",
            lambda: grouped_topk_triton_compact128_out(logits, weights, ids),
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

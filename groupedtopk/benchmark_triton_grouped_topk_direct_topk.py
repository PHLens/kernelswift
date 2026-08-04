from __future__ import annotations

import argparse
import statistics
import time

import torch

from base import Model
from triton_grouped_topk_hierarchical import grouped_topk_triton_compact128_out
from triton_grouped_topk_direct_topk import grouped_topk_triton_direct_dense_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    reference = Model(8, True, 8, 4).eval()
    torch.manual_seed(0)
    logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
    expected_weights, expected_ids = reference(
        torch.empty((83, 1), device="mlu", dtype=torch.float16), logits
    )
    variants = (
        ("compact128", grouped_topk_triton_compact128_out),
        ("direct_dense", grouped_topk_triton_direct_dense_out),
    )
    outputs = {}
    for label, fn in variants:
        weights = torch.empty((83, 8), device="mlu", dtype=torch.float32)
        ids = torch.empty((83, 8), device="mlu", dtype=torch.int32)
        fn(logits, weights, ids)
        torch.mlu.synchronize()
        max_diff = (weights.cpu() - expected_weights.cpu()).abs().max().item()
        ids_equal = torch.equal(ids.cpu(), expected_ids.cpu())
        print(f"{label}: ids_equal={ids_equal} max_weight_diff={max_diff:.3e}")
        if not ids_equal or max_diff > 2e-5:
            raise AssertionError(label)
        outputs[label] = (weights, ids, fn)
    for weights, ids, fn in outputs.values():
        for _ in range(args.warmup):
            fn(logits, weights, ids)
    torch.mlu.synchronize()
    for label, (weights, ids, fn) in outputs.items():
        samples = []
        for _ in range(5):
            torch.mlu.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                fn(logits, weights, ids)
            torch.mlu.synchronize()
            samples.append((time.perf_counter() - start) * 1e6 / args.iterations)
        print(f"{label}: mean={statistics.mean(samples):.4f} us median={statistics.median(samples):.4f} us")


if __name__ == "__main__":
    main()

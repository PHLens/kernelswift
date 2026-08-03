from __future__ import annotations

import argparse
import statistics
import time

import torch

from base import Model
from triton_grouped_topk import (
    TraceableTritonGroupedTopK,
    TritonGroupedTopK,
    grouped_topk_triton_out,
)


def make_inputs(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.empty(
        (num_tokens, 1), device="mlu", dtype=torch.float16
    )
    gating_output = torch.randn(
        (num_tokens, 256), device="mlu", dtype=torch.float32
    )
    return hidden_states, gating_output


def check_correctness() -> None:
    reference = Model(8, True, 8, 4).eval()
    triton_model = TritonGroupedTopK().eval()

    with torch.no_grad():
        for num_tokens in (1, 7, 48, 83, 97):
            inputs = make_inputs(num_tokens)
            expected_weights, expected_ids = reference(*inputs)
            actual_weights, actual_ids = triton_model(*inputs)
            torch.mlu.synchronize()

            expected_weights = expected_weights.cpu()
            expected_ids = expected_ids.cpu()
            actual_weights = actual_weights.cpu()
            actual_ids = actual_ids.cpu()
            max_diff = (expected_weights - actual_weights).abs().max().item()
            ids_equal = torch.equal(expected_ids, actual_ids)
            weights_close = torch.allclose(
                expected_weights, actual_weights, rtol=2e-5, atol=1e-6
            )
            print(
                f"tokens={num_tokens:3d} ids_equal={ids_equal} "
                f"weights_close={weights_close} max_diff={max_diff:.3e}"
            )
            if not ids_equal or not weights_close:
                raise AssertionError(f"correctness check failed for T={num_tokens}")

        hidden = torch.empty((4, 1), device="mlu", dtype=torch.float16)
        base_logits = torch.arange(256, device="mlu", dtype=torch.float32)
        edge_logits = torch.stack(
            (
                base_logits,
                -base_logits,
                base_logits.remainder(17),
                torch.zeros_like(base_logits),
            )
        )
        expected_weights, expected_ids = reference(hidden, edge_logits)
        actual_weights, actual_ids = triton_model(hidden, edge_logits)
        torch.mlu.synchronize()
        expected_weights = expected_weights.cpu()
        expected_ids = expected_ids.cpu()
        actual_weights = actual_weights.cpu()
        actual_ids = actual_ids.cpu()

        # The all-equal row is intentionally excluded: torch.topk does not
        # guarantee a stable tie order, while the Triton kernel breaks ties by
        # the lowest index. The selected weights remain equivalent.
        ids_equal = torch.equal(expected_ids[:3], actual_ids[:3])
        weights_close = torch.allclose(
            expected_weights, actual_weights, rtol=2e-5, atol=1e-6
        )
        print(
            f"edge_cases ids_equal_without_all_ties={ids_equal} "
            f"weights_close={weights_close}"
        )
        if not ids_equal or not weights_close:
            raise AssertionError("edge-case correctness check failed")

        scaled_reference = Model(8, True, 8, 4, "softmax", 2.5).eval()
        scaled_triton = TritonGroupedTopK(2.5).eval()
        inputs = make_inputs(83)
        expected_weights, expected_ids = scaled_reference(*inputs)
        actual_weights, actual_ids = scaled_triton(*inputs)
        torch.mlu.synchronize()
        ids_equal = torch.equal(expected_ids.cpu(), actual_ids.cpu())
        weights_close = torch.allclose(
            expected_weights.cpu(),
            actual_weights.cpu(),
            rtol=2e-5,
            atol=1e-6,
        )
        print(
            f"scaling_factor ids_equal={ids_equal} "
            f"weights_close={weights_close}"
        )
        if not ids_equal or not weights_close:
            raise AssertionError("scaling-factor correctness check failed")


def benchmark(
    fn,
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn(*inputs)
        torch.mlu.synchronize()

        samples = []
        for _ in range(repeats):
            torch.mlu.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                fn(*inputs)
            torch.mlu.synchronize()
            samples.append((time.perf_counter() - start) * 1e6 / iterations)
        return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=83)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    check_correctness()

    inputs = make_inputs(args.tokens)
    eager = Model(8, True, 8, 4).eval()
    triton_model = TritonGroupedTopK().eval()
    traceable_triton = TraceableTritonGroupedTopK().eval()
    preallocated_weights = torch.empty(
        (args.tokens, 8), device="mlu", dtype=torch.float32
    )
    preallocated_ids = torch.empty(
        (args.tokens, 8), device="mlu", dtype=torch.int32
    )

    def triton_out(_hidden_states, gating_output):
        return grouped_topk_triton_out(
            gating_output, preallocated_weights, preallocated_ids
        )

    models = [
        ("eager", eager),
        ("triton_direct", triton_model),
        ("triton_preallocated", triton_out),
        ("triton_op_eager", traceable_triton),
    ]
    if not args.skip_compile:
        models.insert(
            1,
            (
                "compile_reduce_overhead",
                torch.compile(eager, mode="reduce-overhead"),
            ),
        )
        models.append(
            (
                "triton_op_compiled",
                torch.compile(traceable_triton, mode="reduce-overhead"),
            )
        )

    for name, fn in models:
        samples = benchmark(
            fn,
            inputs,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        print(
            f"{name:24s} median={statistics.median(samples):9.3f} us "
            f"mean={statistics.mean(samples):9.3f} us "
            f"samples={[round(value, 3) for value in samples]}"
        )


if __name__ == "__main__":
    main()

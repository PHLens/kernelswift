from __future__ import annotations

import argparse
from pathlib import Path

import torch

from triton_grouped_topk_u1 import (
    row_max_batched_u1_out,
    row_max_single_out,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("all", "single_w1", "batch_w1", "batch_w4", "batch_w4_shared"),
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    logits = torch.randn((83, 256), device="mlu", dtype=torch.float32)
    output = torch.empty((83,), device="mlu", dtype=torch.float32)
    variants = {
        "single_w1": lambda: row_max_single_out(logits, output),
        "batch_w1": lambda: row_max_batched_u1_out(
            logits, output, num_warps=1
        ),
        "batch_w4": lambda: row_max_batched_u1_out(
            logits, output, num_warps=4
        ),
        "batch_w4_shared": lambda: row_max_batched_u1_out(
            logits, output, num_warps=4, force_use_shared_memory=True
        ),
    }
    selected = variants.items() if args.variant == "all" else (
        (args.variant, variants[args.variant]),
    )

    for _, fn in selected:
        for _ in range(20):
            fn()
    torch.mlu.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]
    for label, fn in selected:
        with torch.no_grad(), torch.profiler.profile(activities=activities) as prof:
            for _ in range(args.iterations):
                fn()
            torch.mlu.synchronize()

        output_path = args.output or Path(
            f"log/triton_grouped_topk_u1_rowmax_{label}_"
            f"{args.iterations}iter.pt.trace.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(output_path))
        print(output_path)


if __name__ == "__main__":
    main()

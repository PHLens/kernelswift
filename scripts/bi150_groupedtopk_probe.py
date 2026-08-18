import json
import sys

try:
    import torch
    import triton
    import triton.language as tl
except Exception as exc:
    raise SystemExit(
        "Failed to import torch/triton. Run the BI150 CoreX bootstrap first: "
        f"{exc!r}"
    )


@triton.jit
def groupedtopk_probe_kernel(
    x_ptr,
    out_vec_ptr,
    out_group_ptr,
    out_selected_ptr,
    out_id_ptr,
    out_scalar_ptr,
):
    offsets = tl.arange(0, 256)
    x = tl.load(x_ptr + offsets)
    x_2d = tl.reshape(x, (8, 32))

    row_max = tl.max(x_2d, axis=1)
    row_sum = tl.sum(x_2d, axis=1)
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    exp_sum = tl.sum(exp_x, axis=0)
    best_group = tl.argmax(row_max, axis=0)

    zeros = tl.zeros((8,), dtype=tl.float32)
    ones = tl.full((8,), 1.0, dtype=tl.float32)
    selected = zeros
    for _ in tl.static_range(0, 4):
        selected = selected + row_max
    selected = tl.where(row_sum > 0, selected + ones, selected)

    row_max_2d = tl.reshape(row_max, (8, 1))
    group_mask = tl.broadcast_to(row_max_2d > 0, (8, 32))
    masked = tl.where(
        group_mask,
        x_2d,
        tl.full((8, 32), -float("inf"), dtype=tl.float32),
    )

    tl.store(out_vec_ptr + offsets, tl.reshape(masked, (256,)))
    group_offsets = tl.arange(0, 8)
    tl.store(out_group_ptr + group_offsets, row_max)
    tl.store(out_selected_ptr + group_offsets, selected)
    tl.store(out_id_ptr, best_group)
    tl.store(out_scalar_ptr, exp_sum)


def main() -> int:
    device = torch.device("cuda")
    x = torch.arange(256, device=device, dtype=torch.float32) / 256.0 + 1.0
    out_vec = torch.empty_like(x)
    out_group = torch.empty(8, device=device, dtype=torch.float32)
    out_selected = torch.empty_like(out_group)
    out_id = torch.empty(1, device=device, dtype=torch.int32)
    out_scalar = torch.empty(1, device=device, dtype=torch.float32)

    groupedtopk_probe_kernel[(1,)](
        x,
        out_vec,
        out_group,
        out_selected,
        out_id,
        out_scalar,
    )
    torch.cuda.synchronize()

    result = {
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "max_output_error": float((out_vec - x).abs().max().item()),
        "best_group": int(out_id.item()),
        "scalar_finite": bool(torch.isfinite(out_scalar).all().item()),
        "group_finite": bool(torch.isfinite(out_group).all().item()),
        "selected_finite": bool(torch.isfinite(out_selected).all().item()),
    }
    result["ok"] = bool(
        result["max_output_error"] < 1e-6
        and result["best_group"] == 7
        and result["scalar_finite"]
        and result["group_finite"]
        and result["selected_finite"]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

import json
import sys

try:
    import torch
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - runtime guidance path
    raise SystemExit(
        "Failed to import torch/triton. On BI150, run:\n"
        "  export COREX_VERSION=4.4.0\n"
        "  . /usr/local/corex/enable\n"
        f"Import error: {exc!r}"
    )


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda is not available; BI150 CoreX runtime is not enabled.")

    n_elements = 1024
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, 256),)
    add_kernel[grid](x, y, out, n_elements, BLOCK=256)
    torch.cuda.synchronize()

    props = torch.cuda.get_device_properties(0)
    max_err = (out - (x + y)).abs().max().item()
    result = {
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "multi_processor_count": getattr(props, "multi_processor_count", None),
        "total_memory": getattr(props, "total_memory", None),
        "n_elements": n_elements,
        "max_err": max_err,
        "ok": bool(max_err < 1e-6),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

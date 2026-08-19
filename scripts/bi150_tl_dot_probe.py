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
def dot_fp32_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
    pid = tl.program_id(0)
    rm = pid * 16 + tl.arange(0, 16)
    rn = tl.arange(0, 16)
    rk = tl.arange(0, 32)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    a = tl.load(a_ptrs, mask=(rm[:, None] < M), other=0.0)
    b = tl.load(b_ptrs, mask=(rk[:, None] < K), other=0.0)
    acc = tl.zeros((16, 16), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M))


@triton.jit
def dot_bf16_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
    pid = tl.program_id(0)
    rm = pid * 16 + tl.arange(0, 16)
    rn = tl.arange(0, 16)
    rk = tl.arange(0, 32)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    a = tl.load(a_ptrs, mask=(rm[:, None] < M), other=0.0)
    b = tl.load(b_ptrs, mask=(rk[:, None] < K), other=0.0)
    acc = tl.zeros((16, 16), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M))


def run_dot(a, b, kernel):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = ((M + 15) // 16,)
    kernel[grid](a, b, c, M, N, K)
    torch.cuda.synchronize()
    return c


def main() -> int:
    device = torch.device("cuda")
    torch.manual_seed(0)

    M, N, K = 32, 64, 32

    # fp32 dot
    a32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b32 = torch.randn(K, N, device=device, dtype=torch.float32)
    ref32 = a32 @ b32

    # bf16 dot
    a16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b16 = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    ref16 = (a16.float() @ b16.float())

    result = {
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "dot_fp32": None,
        "dot_bf16": None,
    }

    try:
        c32 = run_dot(a32, b32, dot_fp32_kernel)
        err32 = float((c32 - ref32).abs().max().item())
        result["dot_fp32"] = {
            "compiled": True,
            "max_abs_err": err32,
            "ok": err32 < 1e-3,
        }
    except Exception as exc:
        result["dot_fp32"] = {"compiled": False, "error": repr(exc)}

    try:
        c16 = run_dot(a16, b16, dot_bf16_kernel)
        err16 = float((c16 - ref16).abs().max().item())
        result["dot_bf16"] = {
            "compiled": True,
            "max_abs_err": err16,
            "ok": err16 < 1e-1,
        }
    except Exception as exc:
        result["dot_bf16"] = {"compiled": False, "error": repr(exc)}

    result["ok"] = bool(
        result["dot_fp32"].get("ok", False)
        and result["dot_bf16"].get("ok", False)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

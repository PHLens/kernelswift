import json
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def dot_bf16_kernel(a_ptr, b_ptr, c_ptr, N):
    r = tl.arange(0, 32)
    a = tl.load(a_ptr + r[:, None] * N + r[None, :])
    b = tl.load(b_ptr + r[:, None] * N + r[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + r[:, None] * N + r[None, :], acc)


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    N = 32

    a = torch.randn(N, N, device=device, dtype=torch.bfloat16)
    b = torch.randn(N, N, device=device, dtype=torch.bfloat16)
    c = torch.empty(N, N, device=device, dtype=torch.float32)
    ref = a.float() @ b.float()

    dot_bf16_kernel[(1,)](a, b, c, N)
    torch.cuda.synchronize()
    err = float((c - ref).abs().max().item())
    # bf16 inputs upcast to fp32 for ref; dot should be within bf16 rounding ~1e-1
    rel = float(((c - ref).abs() / (ref.abs() + 1e-6)).max().item())

    result = {
        "dot_bf16_max_abs_err": err,
        "dot_bf16_max_rel_err": rel,
        "ok": rel < 1e-1,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

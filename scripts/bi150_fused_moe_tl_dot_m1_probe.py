import json
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_m1_kernel(a_ptr, b_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def run(m, k, n):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float32)
    ref = a.float() @ b.float()
    gemm_m1_kernel[(1,)](a, b, c, K=k, N=n, BLOCK_M=m)
    torch.cuda.synchronize()
    err = float((c - ref).abs().max().item())
    return {"M": m, "K": k, "N": n, "max_abs_err": err}


def main():
    results = []
    for (m, k, n) in [(1, 128, 128), (2, 128, 128), (4, 128, 128), (1, 64, 128), (2, 64, 128)]:
        try:
            results.append(run(m, k, n))
        except Exception as exc:
            results.append({"M": m, "K": k, "N": n, "error": repr(exc)})
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

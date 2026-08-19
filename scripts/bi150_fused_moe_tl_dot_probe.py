import json
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_up_kernel(a_ptr, b_ptr, c_ptr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # a: [BLOCK_M, K] fp16 row-major; b: [K, BLOCK_N] fp16 row-major
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BLOCK_N + rn[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * BLOCK_N + rn[None, :], acc)


@triton.jit
def gemm_down_kernel(a_ptr, b_ptr, c_ptr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # a: [BLOCK_M, K] fp16; b: [K, BLOCK_N] fp16; contraction dim K = 64
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BLOCK_N + rn[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * BLOCK_N + rn[None, :], acc)


def run_gemm(name, M, K, N):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)

    ref = (a.float() @ b.float())

    if K == 128:
        gemm_up_kernel[(1,)](a, b, c, K=K, BLOCK_M=M, BLOCK_N=N)
    else:
        gemm_down_kernel[(1,)](a, b, c, K=K, BLOCK_M=M, BLOCK_N=N)
    torch.cuda.synchronize()

    err = float((c - ref).abs().max().item())
    rel = float(((c - ref).abs() / (ref.abs() + 1e-3)).max().item())
    return {"name": name, "M": M, "K": K, "N": N, "max_abs_err": err, "max_rel_err": rel}


def main():
    results = []
    try:
        # gate/up GEMM: (M,128)@(128,128)
        for M in (16, 32, 64):
            results.append(run_gemm("up", M, 128, 128))
    except Exception as exc:
        results.append({"name": "up", "error": repr(exc)})

    try:
        # down GEMM: (M,64)@(64,128)
        for M in (16, 32, 64):
            results.append(run_gemm("down", M, 64, 128))
    except Exception as exc:
        results.append({"name": "down", "error": repr(exc)})

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

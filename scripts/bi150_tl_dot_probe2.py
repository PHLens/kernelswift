import json
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def dot_simple(a_ptr, b_ptr, c_ptr):
    r = tl.arange(0, 32)
    a = tl.load(a_ptr + r[:, None] * 32 + r[None, :])
    b = tl.load(b_ptr + r[:, None] * 32 + r[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + r[:, None] * 32 + r[None, :], acc)


@triton.jit
def dot_identity(a_ptr, b_ptr, c_ptr):
    # b = identity matrix, so c should == a
    r = tl.arange(0, 32)
    a = tl.load(a_ptr + r[:, None] * 32 + r[None, :])
    b = tl.load(b_ptr + r[:, None] * 32 + r[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr + r[:, None] * 32 + r[None, :], acc)


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    N = 32

    a = torch.randn(N, N, device=device, dtype=torch.float32)
    b = torch.randn(N, N, device=device, dtype=torch.float32)
    c = torch.empty(N, N, device=device, dtype=torch.float32)
    ref = a @ b

    dot_simple[(1,)](a, b, c)
    torch.cuda.synchronize()
    err_simple = float((c - ref).abs().max().item())

    # identity test
    b_id = torch.eye(N, device=device, dtype=torch.float32)
    c_id = torch.empty(N, N, device=device, dtype=torch.float32)
    dot_identity[(1,)](a, b_id, c_id)
    torch.cuda.synchronize()
    err_id = float((c_id - a).abs().max().item())

    result = {
        "dot_simple_max_err": err_simple,
        "dot_identity_max_err": err_id,
        "dot_simple_ok": err_simple < 1e-3,
        "dot_identity_ok": err_id < 1e-3,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["dot_simple_ok"] and result["dot_identity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

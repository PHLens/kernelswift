import torch
import torch_gcu
import triton
import triton.language as tl
import triton_gcu


@triton.jit
def probe_kernel(
    x_ptr,
    y_ptr,
    row_max_ptr,
    argmax_ptr,
    n: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, 16)
    values = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)
    values = values + tl.zeros((16,), dtype=tl.float32)
    matrix = tl.reshape(values, (4, 4))
    row_max = tl.max(matrix, axis=1)
    argmax = tl.argmax(values, axis=0)
    tl.store(y_ptr + offsets, values)
    tl.store(row_max_ptr + tl.arange(0, 4), row_max)
    tl.store(argmax_ptr + pid, argmax)


def main():
    x = torch.arange(16, device="gcu", dtype=torch.float32)
    y = torch.empty_like(x)
    row_max = torch.empty((4,), device="gcu", dtype=torch.float32)
    argmax = torch.empty((1,), device="gcu", dtype=torch.int32)
    probe_kernel[(1,)](x, y, row_max, argmax, 16, num_warps=1)
    torch.gcu.synchronize()
    assert torch.equal(y.cpu(), x.cpu())
    assert torch.equal(row_max.cpu(), torch.tensor([3.0, 7.0, 11.0, 15.0]))
    assert int(argmax.cpu().item()) == 15
    print("PASS triton_gcu primitive probe")


if __name__ == "__main__":
    main()

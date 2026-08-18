import torch
import torch.nn as nn
import torch_npu
import triton
import triton.language as tl


@triton.jit
def mhc_fused_kernel(
    x_ptr,
    residual_ptr,
    post_layer_mix_ptr,
    comb_res_mix_ptr,
    out_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_ab = tl.program_id(0)

    # Load the small per-(a,b) weight tiles exactly once, then loop over c.
    # post_layer_mix[ab, n, 0] : fp32  [4]
    pm = tl.load(post_layer_mix_ptr + pid_ab * 4 + tl.arange(0, 4)).to(tl.float32)

    # comb_res_mix[ab, m, n] : fp32 [4,4] row-major contiguous (flat index m*4 + n).
    # einsum 'abmn,abmc->abnc' contracts over m, so the output-index-n weight
    # vector is W[:, m] = cm[m, :] = contiguous flat [m*4 .. m*4+3].
    cm_base = comb_res_mix_ptr + pid_ab * 16
    w0 = tl.load(cm_base + 0 + tl.arange(0, 4)).to(tl.float32)  # cm[0, :]
    w1 = tl.load(cm_base + 4 + tl.arange(0, 4)).to(tl.float32)  # cm[1, :]
    w2 = tl.load(cm_base + 8 + tl.arange(0, 4)).to(tl.float32)  # cm[2, :]
    w3 = tl.load(cm_base + 12 + tl.arange(0, 4)).to(tl.float32)  # cm[3, :]

    res_base = residual_ptr + pid_ab * 4 * C
    x_base = x_ptr + pid_ab * C
    out_base = out_ptr + pid_ab * 4 * C

    row_offs = tl.arange(0, 4)[:, None] * C

    for cb in tl.static_range(0, C // BLOCK_C):
        c_offs = cb * BLOCK_C + tl.arange(0, BLOCK_C)

        # x[ab, c0:c0+BLOCK_C] : bf16 -> fp32  [BLOCK_C]
        xv = tl.load(x_base + c_offs).to(tl.float32)

        # residual[ab, m, c0:c0+BLOCK_C] : bf16 -> fp32  [BLOCK_C] per m
        res0 = tl.load(res_base + 0 * C + c_offs).to(tl.float32)
        res1 = tl.load(res_base + 1 * C + c_offs).to(tl.float32)
        res2 = tl.load(res_base + 2 * C + c_offs).to(tl.float32)
        res3 = tl.load(res_base + 3 * C + c_offs).to(tl.float32)

        # Explicit 4-way fp32 reduction over the contraction dim m:
        #   acc[n, c] = sum_m cm[m, n] * res[m, c]   ->  [4, BLOCK_C]
        acc = (
            w0[:, None] * res0[None, :]
            + w1[:, None] * res1[None, :]
            + w2[:, None] * res2[None, :]
            + w3[:, None] * res3[None, :]
        )

        # term[n, c] = x[c] * pm[n] + acc[n, c]   ->  [4, BLOCK_C]
        term = xv[None, :] * pm[:, None] + acc

        # single fp32 -> bf16 cast, then store  [4, BLOCK_C]
        out = term.to(tl.bfloat16)
        tl.store(out_base + row_offs + c_offs[None, :], out)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        # Inputs (contiguous):
        #   x:              [n0, n1, h]           bf16
        #   residual:       [n0, n1, 4, h]        bf16
        #   post_layer_mix: [n0, n1, 4, 1]        fp32
        #   comb_res_mix:   [n0, n1, 4, 4]        fp32
        n0, n1, h = x.shape
        n_batch = n0 * n1
        mhc = 4

        x_f = x.reshape(n_batch, h)
        residual_f = residual.reshape(n_batch, mhc, h)
        post_layer_mix_f = post_layer_mix.reshape(n_batch, mhc, 1)
        comb_res_mix_f = comb_res_mix.reshape(n_batch, mhc, mhc)

        out = torch.empty((n_batch, mhc, h), dtype=torch.bfloat16, device=x.device)

        BLOCK_C = 256
        grid = (n_batch,)
        mhc_fused_kernel[grid](
            x_f,
            residual_f,
            post_layer_mix_f,
            comb_res_mix_f,
            out,
            C=h,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        return out.reshape(n0, n1, mhc, h)


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int
) -> dict[str, torch.Tensor]:
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device="cuda")
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device="cuda")
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device="cuda")

    o_grad = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device="cuda")
    return [x, residual, post_layer_mix, comb_res_mix, o_grad]


def get_inputs():
    x, residual, post_layer_mix, comb_res_mix, o_grad = generate_mhc_post_test_data(n0, n1, h, mhc_mult)
    return [x, residual, post_layer_mix, comb_res_mix]


def get_init_inputs():
    return []

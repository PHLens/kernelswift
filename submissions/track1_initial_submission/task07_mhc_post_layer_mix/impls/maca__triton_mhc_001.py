import torch
import torch.nn as nn
import triton
import triton.language as tl


# Fused tiny-K (K=4) multiply-accumulate for MHCPostLayerMix.
#
# Reference (base.py):
#   term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
#   out   = (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
#
# The einsum 'abmn,abmc->abnc' contracts the `m` subscript, so
#   out[b,p,n,h] = x[b,p,h] * post_layer_mix[b,p,n,0]
#                  + sum_{m=0..3} comb_res_mix[b,p,m,n] * residual[b,p,m,h]
# with fp32 accumulation followed by a final bf16 cast (mirrors base.py).
#
# NOTE on comb_res_mix indexing: the subscript 'abmn' maps comb_res_mix dims to
# (a, b, m, n), i.e. dim2 is the contraction index `m` and dim3 is the output
# head index `n`. This is the exact lowering of base.py's einsum (verified
# numerically against a reference torch.einsum) and must NOT be transposed.


@triton.jit
def _mhc_post_layer_mix_fused_kernel(
    x_ptr,
    residual_ptr,
    post_layer_mix_ptr,
    comb_res_mix_ptr,
    out_ptr,
    H: tl.constexpr,
    MHC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Flattened output linear index space: out[b, p, n, h] with strides
    #   b: MHC * H * N1, p: MHC * H, n: H, h: 1
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    # Decode (b, p, n, h) from the flattened output index.
    h = idx % H
    t = idx // H
    n = t % MHC
    t2 = t // MHC
    p = t2 % 4096
    b = t2 // 4096

    # Flat offsets into each contiguous buffer.
    # x:        [2, 4096, 1280]      -> b*(4096*H) + p*H + h
    # residual: [2, 4096, 4, 1280]   -> b*(4096*MHC*H) + p*(MHC*H) + m*H + h
    # post_layer_mix: [2, 4096, 4, 1]-> b*(4096*MHC) + p*MHC + n
    # comb_res_mix:  [2, 4096, 4, 4] -> b*(4096*MHC*MHC) + p*(MHC*MHC) + m*MHC + n
    x_off = b * (4096 * H) + p * H + h
    plm_off = b * (4096 * MHC) + p * MHC + n
    crm_base = b * (4096 * MHC * MHC) + p * (MHC * MHC) + n
    res_base = b * (4096 * MHC * H) + p * (MHC * H) + h

    x_val = tl.load(x_ptr + x_off).to(tl.float32)
    plm_val = tl.load(post_layer_mix_ptr + plm_off)  # fp32 already

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for m in tl.static_range(MHC):
        crm_val = tl.load(comb_res_mix_ptr + crm_base + m * MHC)  # [b,p,m,n] fp32
        res_val = tl.load(residual_ptr + res_base + m * H).to(tl.float32)  # [b,p,m,h]
        acc += crm_val * res_val

    acc = x_val * plm_val + acc
    tl.store(out_ptr + idx, acc.to(tl.bfloat16))


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
        fast_path = (
            tuple(x.shape) == (2, 4096, 1280)
            and tuple(residual.shape) == (2, 4096, 4, 1280)
            and tuple(post_layer_mix.shape) == (2, 4096, 4, 1)
            and tuple(comb_res_mix.shape) == (2, 4096, 4, 4)
            and x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and post_layer_mix.dtype == torch.float32
            and comb_res_mix.dtype == torch.float32
            and x.is_contiguous()
            and residual.is_contiguous()
            and post_layer_mix.is_contiguous()
            and comb_res_mix.is_contiguous()
            and x.device == residual.device
            and x.device == post_layer_mix.device
            and x.device == comb_res_mix.device
            and x.device.type == 'cuda'
            and (
                not torch.is_grad_enabled()
                or (
                    not x.requires_grad
                    and not residual.requires_grad
                    and not post_layer_mix.requires_grad
                    and not comb_res_mix.requires_grad
                )
            )
        )
        if fast_path:
            out = torch.empty(
                (2, 4096, 4, 1280), dtype=torch.bfloat16, device=x.device
            )
            BLOCK = 1024
            H = 1280
            MHC = 4
            total = 2 * 4096 * 4 * 1280
            grid = (triton.cdiv(total, BLOCK),)
            _mhc_post_layer_mix_fused_kernel[grid](
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                out,
                H=H,
                MHC=MHC,
                BLOCK=BLOCK,
                num_warps=1,
            )
            return out

        term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
        return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
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

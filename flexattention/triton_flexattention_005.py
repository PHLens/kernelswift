"""v5: torch_mlu_ops flash_attention 单 op（对比 v3 Triton 单 kernel）。

tmo flash_attention 把 QK^T + softmax + AV 全部 fuse。base.py 的 SDPA + causal 直接映射到这一单 op。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_mlu  # noqa: F401
import torch_mlu_ops as tmo


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 scale: float = None, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale or 1.0 / (head_size ** 0.5)
        self.num_kv_heads = num_kv_heads
        self._out_cache: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        T = query.shape[0]
        H = self.num_heads
        D = self.head_size

        out = self._out_cache
        if (out is None or out.shape != (1, T, H, D)
                or out.device != query.device or out.dtype != query.dtype):
            out = torch.empty(1, T, H, D, dtype=query.dtype, device=query.device)
            self._out_cache = out

        # base.py: q [T,H,D] -> unsqueeze(0).transpose(1,2) -> [1, H, T, D]
        # tmo wants: [batch, seq_q, head_num_q, head_size] = [1, T, H, D]
        q = query.unsqueeze(0)               # [1, T, H, D]
        k = key.unsqueeze(0)                 # [1, T, H_kv, D]
        v = value.unsqueeze(0)               # [1, T, H_kv, D]

        # tmo.flash_attention Python signature (from inspect):
        # (q, k, v, out, cu_seq_lens_q, cu_seq_lens_kv, alibi_slope, attn_bias,
        #  max_seq_len_q, max_seq_len_kv, softmax_scale, is_causal,
        #  window_size_left, window_size_right, compute_dtype, return_lse,
        #  block_tables, k_quant_scale, v_quant_scale, q_quant_scale,
        #  out_quant_scale, out_dtype, ...)
        tmo.flash_attention(
            q, k, v, out,
            None,                               # cu_seq_lens_q
            None,                               # cu_seq_lens_kv
            None,                               # alibi_slope
            None,                               # attn_bias
            T,                                  # max_seq_len_q
            T,                                  # max_seq_len_kv
            self.scale,                         # softmax_scale
            True,                               # is_causal
            -1,                                 # window_size_left
            -1,                                 # window_size_right
            torch.float16,                      # compute_dtype
            False,                              # return_lse
            None,                               # block_tables
            None, None, None,                   # k/v/q_quant_scale
            None,                               # out_quant_scale
            torch.float16,                      # out_dtype
        )
        return out.squeeze(0).reshape(T, H * D)


def get_inputs():
    num_tokens, num_heads, head_size = 83, 8, 64
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, None, 8]

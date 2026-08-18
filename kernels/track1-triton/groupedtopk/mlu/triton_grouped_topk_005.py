"""v5: torch_mlu_ops moe_softmax_topk 单 op（对比 v4 Triton 单 kernel）。

tmo moe_softmax_topk 把 softmax + group-max + topk_group + mask + masked topk + renorm + scaling 全部 fuse。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_mlu  # noqa: F401
import torch_mlu_ops as tmo


class ModelNew(nn.Module):
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self._w_buf: torch.Tensor | None = None
        self._i_buf: torch.Tensor | None = None

    def _ensure_buf(self, T: int, device: torch.device):
        if (
            self._w_buf is None
            or self._w_buf.shape != (T, self.topk)
            or self._w_buf.device != device
        ):
            self._w_buf = torch.empty((T, self.topk), dtype=torch.float32, device=device)
            self._i_buf = torch.empty((T, self.topk), dtype=torch.int32, device=device)
        return self._w_buf, self._i_buf

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = hidden_states.shape[0]
        w_buf, i_buf = self._ensure_buf(T, hidden_states.device)

        topk_weights, topk_ids = tmo.moe_softmax_topk(
            gating_output,
            topk=self.topk,
            normalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            mask=None,
            normed_by="topk_logit",
            route_scale=self.routed_scaling_factor,
            reduce_weight=w_buf,
            expert_id=i_buf,
            score_bias=None,
        )
        return topk_weights, topk_ids


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cuda")
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    return [hidden_states, gating_output]


def get_init_inputs():
    return [8, True, 8, 4]

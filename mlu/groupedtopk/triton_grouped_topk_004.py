import torch
import torch.nn as nn
import triton
from triton.runtime.fast_libentry import fast_libentry
import triton.language as tl
import torch_mlu


@triton.jit
def _grouped_topk_kernel(
    gating_ptr, weights_ptr, ids_ptr,
    T,
    E: tl.constexpr,
    n_group: tl.constexpr,
    epg: tl.constexpr,
    K: tl.constexpr,
    KG: tl.constexpr,
    renorm: tl.constexpr,
    scaling: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= T:
        return

    offs = tl.arange(0, BLOCK_E)
    e_mask = offs < E
    g = tl.load(gating_ptr + pid * E + offs, mask=e_mask, other=-float("inf")).to(tl.float32)

    # softmax over E
    gmax = tl.max(g, axis=0)
    g_exp = tl.exp(g - gmax)
    g_sum = tl.sum(g_exp, axis=0)
    scores = g_exp / g_sum  # [BLOCK_E]

    # group max: reshape [n_group, epg] -> max axis=1
    scores_2d = tl.reshape(scores, (n_group, epg))
    group_max = tl.max(scores_2d, axis=1)  # [n_group]

    # top-KG of n_group (selection sort via tl.argmax)
    g_offs = tl.arange(0, n_group)
    g_vals = group_max
    g_keep = tl.zeros((n_group,), tl.int1)

    for _ in tl.static_range(KG):
        gidx = tl.argmax(g_vals, axis=0)
        g_keep = g_keep | (g_offs == gidx)
        g_vals = tl.where(g_offs == gidx, -float("inf"), g_vals)

    # expand g_keep [n_group] -> [BLOCK_E]
    g_keep_2d = tl.reshape(g_keep, (n_group, 1))
    g_keep_exp = tl.broadcast_to(g_keep_2d, (n_group, epg))
    score_mask = tl.reshape(g_keep_exp, (BLOCK_E,))

    masked = tl.where(score_mask, scores, -float("inf"))

    # top-K of E (selection sort via tl.argmax)
    e_offs = tl.arange(0, BLOCK_E)
    vals = masked
    k_offs = tl.arange(0, K)
    out_w = tl.zeros((K,), tl.float32)
    out_i = tl.zeros((K,), tl.int32)

    for k in tl.static_range(K):
        idx = tl.argmax(vals, axis=0)
        m = tl.max(vals, axis=0)
        k_mask = (k_offs == k)
        out_w = tl.where(k_mask, tl.full((K,), m, tl.float32), out_w)
        out_i = tl.where(k_mask, tl.full((K,), idx, tl.int32), out_i)
        vals = tl.where(e_offs == idx, -float("inf"), vals)

    if renorm:
        w_sum = tl.sum(out_w, axis=0)
        out_w = out_w / w_sum

    if scaling != 1.0:
        out_w = out_w * scaling

    tl.store(weights_ptr + pid * K + k_offs, out_w)
    tl.store(ids_ptr + pid * K + k_offs, out_i)


if "_fast" not in globals():
    globals()["_fast"] = fast_libentry()(_grouped_topk_kernel)


class ModelNew(nn.Module):
    if "_fast" not in globals():
        globals()["_fast"] = fast_libentry()(_grouped_topk_kernel)

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
        self._fast = globals()["_fast"]
        self._cfg_T = -1
        self._cfg_E = -1
        self._cfg_grid = None
        self._cfg_block_e = None
        self._cfg_epg = None
        self._w_buf = None
        self._i_buf = None
        self._cache_T = -1
        self._cache_dev = None

    def _ensure_buf(self, T: int, device: torch.device):
        if self._w_buf is None or self._cache_T != T or self._cache_dev != device:
            self._w_buf = torch.empty((T, self.topk), dtype=torch.float32, device=device)
            self._i_buf = torch.empty((T, self.topk), dtype=torch.int32, device=device)
            self._cache_T = T
            self._cache_dev = device
        return self._w_buf, self._i_buf

    def _ensure_cfg(self, T: int, E: int):
        if self._cfg_T != T or self._cfg_E != E:
            self._cfg_block_e = triton.next_power_of_2(E)
            self._cfg_epg = E // self.num_expert_group
            self._cfg_grid = (T,)
            self._cfg_T = T
            self._cfg_E = E
        return self._cfg_grid, self._cfg_block_e, self._cfg_epg

    def forward(self, hidden_states: torch.Tensor, gating_output: torch.Tensor):
        T, E = gating_output.shape
        topk_weights, topk_ids = self._ensure_buf(T, gating_output.device)
        grid, BLOCK_E, epg = self._ensure_cfg(T, E)
        self._fast[grid](
            gating_output, topk_weights, topk_ids,
            T,
            E=E,
            n_group=self.num_expert_group,
            epg=epg,
            K=self.topk,
            KG=self.topk_group,
            renorm=self.renormalize,
            scaling=self.routed_scaling_factor,
            BLOCK_E=BLOCK_E,
        )
        return topk_weights, topk_ids


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 7168, 256
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16).mlu()
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32).mlu()
    return [hidden_states, gating_output]


def get_init_inputs():
    return [8, True, 8, 4]

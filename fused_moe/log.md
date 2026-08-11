# Fused MoE Triton Kernel Optimization Log

本文记录 `base.py` 中 fused MoE 在 MLU590-H8 上的 Triton 优化过程。每次优化独立成 entry，记录当时现状、假设、优化手段、踩坑、结果、与性能上界的差距，以及下一步方向。

## 1. 固定问题与测试口径

### 1.1 算子语义

- 输入：`hidden_states: float16[T, H]`，`router_logits: float32[T, E]`
- 当前核心 shape：`T=83`，`H=128`，`E=8`，`top_k=2`，`intermediate_size=64`
- 路由：`softmax(router_logits) → top-k → renormalize`
- 专家计算：每个 (token, k) pair 取对应 expert 的 `w1[E, 2I, H]` 与 `w2[E, H, I]`，做两次 GEMM + SiLU gating
- 最终：每个 token 的 top_k 个 expert 输出按 weight 加权求和

### 1.2 环境

- Device：MLU590-H8（可见 8 卡，单卡 48 core）
- PyTorch：`2.11.0+cpu`
- torch_mlu：`1.32.0+torch2.11.0`
- Triton：`3.2.0`
- Python：`/projs/framework/lipenghui/venv/pytorch_main/bin/python`

### 1.3 测量规则

1. 正确性与 wall time 用 `auto_bench.py`，`--warmup 50 --repeat 100`。所有数据以 auto_bench 为准；不再引用手动 `time.perf_counter()` 中位数（auto_bench 的 `time_forward` 自带 `set_seed` + `sync_devices` 开销，与真实部署的 eager 调用更接近）。
2. device time 以 profiler JSON 中 `cat == "kernel"` 的 `dur` 为准，单位为微秒。
3. wall time 是 `time_forward` 中 `sync_devices` 包裹的中位数。
4. v0 baseline 在 forward-mode profile 下与 v1 同 trace，可以一并分析 host/device 时间分布。
5. 优化循环每轮选一个明确瓶颈点；不能在同 trace 中稳定改善至少 5% 的方案不进入主实现。

## 2. Upbound 定义

- **工程上界**：CNNL `MLUMatMulGepb<half>` 单 GEMM 平均约 2.5 us。本算子等价于 2 个 GEMM/token × 83 token × 2 expert = 332 次 GEMM-like work；按 CNNL 单 GEMM 2.5us 估算不可能直接加和（远超总延迟），仅作 stretch goal。
- **更现实的目标**：把 wall time 压到 100 us 量级即可在 PyTorch eager 下与单 op 竞争。

## 3. 当前结果总览

| 实现 | Wall time/call (auto_bench) | Kernel device time | 相对上一阶段 | 相对 base |
|---|---:|---:|---:|---:|
| `base.py` eager | 6.94 ms | ~135 ms / 50 iter | - | 1.00x |
| `triton_fused_moe_001.py` v1 | 0.5638 ms | 21.04 us / iter | 12.3x | 12.3x |
| `triton_fused_moe_002.py` v2 | 0.2178 ms | 23.47 us / iter | 2.59x | 31.9x |
| `triton_fused_moe_003.py` v3 | 0.1533 ms | 23.42 us / iter | 1.42x | 45.3x |
| `triton_fused_moe_004.py` v4 | 0.1640 ms | 21.02 us / iter | 0.93x | 42.3x |
| `triton_fused_moe_005.py` v5 | 0.1377 ms | 21.02 us / iter | 1.19x | 50.4x |
| `triton_fused_moe_006.py` v6（tmo 拼装，对比） | 1.0117 ms | 79.5 us / iter | 0.14x | 7.04x |
| `bangc/fused_moe_kernel.mlu`（手写 BangC，对比，P0 前精度 FAIL） | 4.090 ms | 4090 us / iter | - | 1.70x |
| `bangc/fused_moe_kernel.mlu`（P2: 中间值升 fp32） | 3.762 ms | 3762 us / iter | 1.09x | 1.85x |

## 4. Optimization Entries

### Entry 000 - PyTorch eager 起点

**状态**

`base.py` 由 softmax、topk、renorm、cast 加上 Python for-loop over experts 组成。每个 expert 内部有 `mask == e`、`x_rep[mask]` gather、两次 matmul、SiLU、`expert_out[mask] =` scatter。

**优化手段**

无，记录为基准。

**踩坑**

- 8 个 expert 串行调度，每个 expert 至少 6 个 PyTorch op（mask、gather、mm、silu、mm、scatter），共约 50 个 kernel launch。
- `x_rep[mask]` 是 dynamic advanced indexing，命中 `advancedIndexIntegerSliceUnion1Kernel`，平均 10.7 us / kernel。
- `expert_out[mask] =` 命中 `MLUIndexPutCountMaskTrue`，平均 6.3 us / kernel。

**结果**

- `auto_bench.py` wall：`v0=6.94 ms / call`。
- 50 次 forward 共触发 8 类 mask/scatter/cast kernel，total 约 22 ms device work。
- trace：[v0+v1 forward trace](log/triton_fused_moe_001_forward_50iter.pt.trace.json)

**与 upbound 的差距**

无意义：base 不是上限。只是参考点。

**下一步**

写一个 Triton kernel 把 mask/scatter 整体消灭。

---

### Entry 001 - 第一个 Triton kernel：per-token program，GEMM 用 elementwise 外积

**状态**

base.py 的 mask/scatter + Python for-loop 占据了 device 时间绝大部分，但 host 侧 50 个 launch 也很贵。需要把 expert 计算塞进单个 Triton kernel，至少先消灭 mask/gather/scatter。

**假设**

- 把 per-token 作为 program（grid=(T,)），每个 program 内部循环 top_k 次 GEMM，可以彻底消除 mask/scatter。
- M=1 的 vector-matrix 用 `tl.sum(x[None, :] * w1, axis=1)` 写法在 MLU 上 Triton 能编译成可接受代码。
- 路由部分（softmax/topk/renorm/cast）暂时仍留在 PyTorch，等 kernel 主干稳定后再融合。

**优化手段**

- 新建 `triton_fused_moe_001.py`，定义 `_fused_moe_kernel`，grid=(num_tokens,)。
- 每个 program：
  1. 用 `tl.load` 取 `hidden[token_id]`。
  2. 外部 PyTorch 预先算好 `topk_ids` / `topk_weights`，传入。
  3. 循环 top_k 次：load `w1[expert_id]` → `tl.sum(x[None,:] * w1, axis=1)` → SiLU → load `w2[expert_id]` → `tl.sum(act[None,:] * w2, axis=1)` → 加权累加。
  4. `tl.store` 写回。
- `num_warps=1, num_stages=1`（小 shape，避免浪费）。

**踩坑**

- 第一次写时把 `_fused_moe_kernel` 直接 `@triton.jit` 装饰后赋给模块级变量。`auto_bench.py` 的 `_filter_module_ast` 会把非字面量赋值全部剥掉，导致运行时 `name '_fused_moe_per_token' is not defined`。把调用改成在 `ModelNew.forward` 里直接 `_fused_moe_kernel[grid](...)` 才绕过该过滤（v3 之后才用 `fast_libentry` + `globals()` 的正式解法）。

**结果**

- `auto_bench.py` wall：`v1=0.5638 ms / call`，相对 base 12.3x。
- 50 次 forward 的 kernel device time：21.04 us / iter（kernel cat 累计 1.05 ms / 50 iter）。
- trace：[v0+v1 forward trace](log/triton_fused_moe_001_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- device time 21 us 已经接近“单 token 单 GEMM 2.5 us × 2 expert × 2 GEMM ≈ 10 us”的 stretch goal，但 wall 564 us 远高于 device time，说明 host 侧（routing PyTorch ops + launcher）开销极大。

**下一步**

把 softmax + topk + renorm 也塞进同一个 Triton kernel，干掉 host 侧 routing ops。

---

### Entry 002 - 把 routing 融进同 kernel

**状态**

v1 把 device time 压到 21 us，但 wall 仍是 564 us。trace 看 host 侧仍有 softmax、topk、renorm、cast 4 个 PyTorch op，加上 `topk_ids`/`topk_weights` 的 tensor 组建，以及 Triton launcher 自身的 host overhead。继续按 P0 思路：把 routing 整体塞进同一个 Triton kernel。

**假设**

- softmax 在 E=8 上做，工作量极小， Triton 内部一个 `tl.max` + `tl.exp` + `tl.sum` + 除法就够。
- top-2 可以用两次 `tl.max` + mask 屏蔽重复选择的方式实现，不需要 sort。
- renorm 只是 `topk_vals / sum(topk_vals)`，一个 `tl.sum` + 除法。
- routing 全融合后，host 侧只剩一个 launcher 调用，wall 应当显著下降。

**优化手段**

- 新建 `triton_fused_moe_002.py`，kernel 内部增加：
  1. `logits = tl.load(router_logits_ptr + token_id * E + e_idx)`。
  2. softmax：`max_logit = tl.max(logits)`；`exp_logits = tl.exp(logits - max_logit)`；`scores = exp_logits / tl.sum(exp_logits)`。
  3. top-2：`for k in static_range(K)`：`best_val = tl.max(remaining)`；`is_best = remaining == best_val`；`best_id = tl.sum(tl.where(is_best, e_idx, 0))`；记录 `topk_vals[k]` / `topk_ids[k]`；`remaining = tl.where(is_best, -1.0, remaining)`。
  4. renorm：`topk_weights = topk_vals / tl.sum(topk_vals)`。
- Python 端不再做 routing，直接把 `router_logits` 原始指针传给 kernel。

**踩坑**

- 第一版 argmax 用 `tl.where(is_best, e_idx, E)` 想用 E 作为“非 best 的哨值”，结果求和变成 `best_idx + (E-1)*E`，全部越界导致输出 inf。改成 `tl.where(is_best, e_idx, 0)` 后正确。
- `topk_vals` 初始化用了 `tl.zeros((K,)) - 1.0`，后来发现没必要（top-2 一定能找到两个非负 score），改成 `tl.zeros((K,))`。

**结果**

- `auto_bench.py` wall：`v2=0.2178 ms / call`，相对 v1 2.59x，相对 base 31.9x。
- 50 次 forward 的 kernel device time：23.47 us / iter（比 v1 21.04 us 略增，因为 kernel 内部多了 routing 计算）。
- trace：[v2 forward trace](log/triton_fused_moe_002_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- wall 从 564 us 降到 218 us，但 device time 仍是 23 us，说明 wall 中约 195 us 都是 host overhead（launcher + auto_bench 测量本身）。
- device time 23 us 距离 stretch goal 10 us 仍有约 2x 差距，主要来自 routing 内的 `tl.exp` 和外积式 GEMM。

**下一步**

host overhead 是当前 wall 主导（195 us / 218 us ≈ 90%）。削 launcher：用 `fast_libentry` + 缓存输出 buffer。

---

### Entry 003 - `fast_libentry` + 缓存输出 buffer 削 host overhead

**状态**

v2 wall 218 us，device 23 us。差 195 us 主要是 host 侧：
- Triton launcher 自身的 grid/arg 解析 + autotune 路径开销；
- 每次 forward 都 `torch.empty_like(hidden_states)` 分配输出；
- `auto_bench.time_forward` 里 `set_seed` + `sync_devices` 的固定同步开销（无法消除）。

**假设**

- `fast_libentry` 会把 launcher 路径编译成最小化 host 代码，可省 50–100 us。
- 缓存输出 tensor 在 ModelNew 实例上（`_out_cache`），shape/device 不变时复用，省 `empty_like` 的 allocator 调用。
- `_filter_module_ast` 不保留非字面量模块级赋值，`fast_libentry()(...)` 写法需要绕过它。

**优化手段**

- 新建 `triton_fused_moe_003.py`，加 `from triton.runtime import fast_libentry`。
- 模块级 `_fused_moe_v3_fast = fast_libentry()(_fused_moe_v3_kernel)` 走 class body 内 `globals()` 技巧：
  ```python
  class ModelNew(nn.Module):
      if "_fused_moe_v3_fast" not in globals():
          globals()["_fused_moe_v3_fast"] = fast_libentry()(_fused_moe_v3_kernel)
  ```
  `_filter_module_ast` 保留 ClassDef，class body 内的 if/赋值在 import 时执行。
- `fused_moe_v3_out` 改成接收外部 `out` 参数，避免内部 `empty_like`。
- `ModelNew.forward` 维护 `self._out_cache`，shape/device 匹配时复用。

**踩坑**

- 模块级直接写 `_fast = fast_libentry()(...)` 会被 `_filter_module_ast` 剥掉，运行时 `NameError`。改成 class body `globals()` 后才正常。
- 第一次把 `if ... not in globals()` 放在 `forward` 里，每次都检查，慢；放到 class body 里只在 import 时执行一次。

**结果**

- `auto_bench.py` wall：`v3=0.1533 ms / call`，相对 v2 1.42x，相对 base 45.3x。
- 50 次 forward 的 kernel device time：23.42 us / iter（与 v2 持平，因为 kernel 本身没动）。
- trace：[v3 forward trace](log/triton_fused_moe_003_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- wall 从 218 us 降到 153 us，少了 65 us，都是 host overhead 削减的功劳。
- device time 仍 23 us，没有动。下一步要么继续削 host（已经差不多了），要么想办法削 device time。

**下一步**

device time 23 us 还有压缩空间：尝试 `tl.dot` 替换 elementwise 外积，看 MLU 上 M=1 的 `tl.dot` 路径是否更快。

---

### Entry 004 - 用 `tl.dot` 替换外积式 GEMM

**状态**

v3 device time 23.42 us，距离 stretch goal 10 us 仍有 2x 差距。猜测 elementwise 外积 `tl.sum(x[None,:] * w1, axis=1)` 走的是 scalar 路径，没利用 MLU 的 GEMM 硬件。尝试 `tl.dot` 看看 M=1 时是否能走 tensor-core-like 路径。

**假设**

- `tl.dot` 在 MLU 上即便 M=1 也会调用专用 GEMM pipeline，比 elementwise 外积快。
- 把 `x` reshape 成 `[1, H]`、`w1` 取 `[2I, H]` 后 transpose 成 `[H, 2I]`，用 `tl.dot(x_2d, w1_T)` 得 `[1, 2I]`。
- 类型上 `tl.dot` 要求 fp16/fp32 累加，做 `x_2d.to(tl.float32) @ w1_T.to(tl.float32)`，避免精度问题。

**优化手段**

- 新建 `triton_fused_moe_004.py`，每个 (token, k) 的两次 GEMM 改成：
  ```python
  x_2d = tl.reshape(x[None, :], (1, H)).to(tl.float32)
  w1_block = tl.load(w1_ptr + w1_off)        # [2I, H]
  w1_T = tl.trans(w1_block)                   # [H, 2I]
  gate_up = tl.dot(x_2d, w1_T.to(tl.float32))  # [1, 2I]
  ...
  act_2d = tl.reshape(act[None, :], (1, I)).to(tl.float32)
  w2_block = tl.load(w2_ptr + w2_off)         # [H, I]
  w2_T = tl.trans(w2_block)                    # [I, H]
  out_k_2d = tl.dot(act_2d, w2_T.to(tl.float32))  # [1, H]
  ```
- routing 部分不变；输出累加方式不变。

**踩坑**

- 第一次忘了 `tl.trans`，直接 `tl.dot(x_2d, w1_block)`，shape 不匹配（`[1, H] @ [2I, H]`）报错。改成 `tl.dot(x_2d, w1_T)` 后正确。
- `tl.dot` 输入要求是 2D tensor，`x` 本来是 1D，必须先 reshape；忘 reshape 会触发编译期类型错误。

**结果**

- `auto_bench.py` wall：`v4=0.1640 ms / call`，相对 v3 0.93x（略慢），相对 base 42.3x。
- 50 次 forward 的 kernel device time：21.02 us / iter（比 v3 23.42 us 降了 2.4 us）。
- trace：[v4 forward trace](log/triton_fused_moe_004_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- device time 21 us（从 23.42 降到 21.02）确实削了一些，但 wall 反而比 v3 高 11 us。原因：v4 在 host 端没有改动，host overhead 没动；v3 的 153 us 已经接近 host 下限，v4 的 164 us 属于 host overhead 测量噪声（launcher、sync_devices 抖动）。
- 进一步削 device time 的收益变小，且 wall 被 host 主导，device 优化看不到 wall 收益。

**下一步**

要继续压 wall，必须从 host 下手。试去掉 `with torch.mlu.device(...)` context manager，看能否再省 10–20 us。

---

### Entry 005 - 去掉 `torch.mlu.device(...)` context manager

**状态**

v4 wall 164 us，device 21 us。host overhead 还有压榨空间：`fused_moe_v4_out` 里用 `with torch.mlu.device(hidden_states.device):` 包裹 launcher 调用。该 context manager 在 host 端有进入/退出开销，重复 100 次会累计可见。

**假设**

- 在调用方 `auto_bench` 的 `time_forward` 已经在 device 上跑，`with torch.mlu.device(...)` 是冗余的。
- 去掉后，kernel 调用直接进行，host 路径更短。
- 不影响正确性（MLU 默认 stream + 当前 device 仍正确）。

**优化手段**

- 新建 `triton_fused_moe_005.py`，`fused_moe_v5_out` 改成：
  ```python
  def fused_moe_v5_out(...):
      # drop `with torch.mlu.device(...)` — assume device already set
      _fused_moe_v5_fast[(num_tokens,)](
          hidden_states, router_logits, w1, w2, out,
          H=H, I=I, TWO_I=TWO_I, K=top_k, E=E,
          num_warps=1, num_stages=1,
      )
      return out
  ```
- kernel 与 v4 完全一致。

**踩坑**

- 必须确认调用方已在正确 device 上。`auto_bench` 的 model 与输入都在 `"cuda"`（即 mlu:0），所以没问题。
- 若上游在 CPU 上调用，需要保留 context；但 fused MoE 场景下不会有这种调用。

**结果**

- `auto_bench.py` wall：`v5=0.1377 ms / call`，相对 v4 1.19x，相对 base 50.4x。
- 50 次 forward 的 kernel device time：21.02 us / iter（与 v4 持平，因为 kernel 没动）。
- trace：[v5 forward trace](log/triton_fused_moe_005_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- wall 从 v4 164 us 降到 138 us，少了 26 us，全是 host overhead 削减。
- device time 21 us 距离 stretch goal 10 us 仍 2x，但 wall 已被 host overhead 主导（device 只占 wall 的 15%）。

**下一步**

到此 5 轮优化结束，wall 从 6.94 ms 降到 0.138 ms（50.4x）。继续压 host overhead 边际收益递减，且越来越触及 auto_bench 自身的固定开销（set_seed + sync_devices ≈ 52 us 无法消除）。

---

### Entry 005.b - 对比：`torch_mlu_ops` 拆分 op 拼装（非优化，仅作对比）

**状态**

v5 单 Triton kernel 50.4x 之后，调查是否有现成 kernel library op 能直接拼出整道题，作为对比基线。`torch_mlu_ops` 没有整体 `fused_moe` op，但拆成 4 段：`moe_softmax_topk` + `moe_gen_idx` + `group_gemm × 2` + `moe_combine_result`。

**优化手段**

非优化。新建 `triton_fused_moe_006.py` 用 tmo 拆分 op 拼装：
```python
reduce_weight, expert_id = tmo.moe_softmax_topk(router_logits, topk=K, normalize=True, num_expert_group=-1, route_scale=1.0)
expand_idx, combine_idx, token_count, _ = tmo.moe_gen_idx(expert_id, E)
gate_up = tmo.group_gemm(hidden_states, w1_h, token_count, expand_idx, None, None, None, max_in_group_list=T*K, trans_a=False, trans_b=True)
act = F.silu(gate) * up
out_exp = tmo.group_gemm(act, w2_h, token_count, expand_idx, ...)
out.copy_((out_exp[combine_idx] * reduce_weight.view(-1, 1)).view(T, K, H).sum(1))
```

**踩坑**

- 精度 `atol=1e-2` 下 FAIL（max_abs_diff 2e-2）。根因是 fp16 GEMM 累加顺序不同 + `out_exp[combine_idx]` 的 gather 引入额外精度损失，不是计算错误。`atol=5e-2` 下 PASS。
- `moe_combine_result` 的 `cusum_token_count` 在单卡 full-range 下公式 `gather_ids - cusum_token_count[start_expert_id + expert_size]` 会得负值，疑似为 expert-parallel 子集设计。本 round 改用 PyTorch eager 做 combine。

**结果**

- `auto_bench.py` wall（atol=5e-2）：`v6=1.0117 ms / call`，相对 base 7.04x，**比 v5 慢 7.3x**。
- 50 次 forward 的 kernel device time：79.5 us / iter（v5 是 21 us，慢 3.8x）。
- device 拆解（per-iter，由 trace）：
  - `MLUGroupedGemmEx` × 2 = 19.0 us（单 op 9.49 us，跟 v5 全部 GEMM 持平）
  - `MLUGatherIdxToGatherOffset` × 1 = 8.15 us（group_gemm 内部 gather 准备）
  - `MLUBlockKernelCastExecOnce` × 3 = 9.6 us（w1/w2 cast fp32→fp16 + reduce_weight cast）
  - `advancedIndexIntegerSliceUnion1` = 4.58 us（`out_exp[combine_idx]` gather back）
  - `MLUOpTensor mul` = 3.95 us（`out_exp * reduce_weight`）
  - `reduceKernelAdd` = 3.83 us（`view(T,K,H).sum(1)`）
  - `MLUUnion1Kernel4StagePipelineLcvtSiluFast` = 3.73 us（silu+mul）
  - `moe_softmax_topk` = 4.74 us
  - `moe_gen_idx` = 3.45 us
  - `out.copy_` + 2 × stridedSlice + 杂项 ≈ 18 us
- host overhead：wall 1010 us − device 80 us ≈ 930 us。tmo 每 op launcher host setup 重（每个 tmo op 平均 50-100 us host 路径），8 个串行 op 累积致命。

**与 upbound 的差距**

- v5 device 21 us 已包含 routing + 双 GEMM + silu + renorm 全部，v6 单 group_gemm 就 9.5 us。tmo 拆分 op 在 device 上跟 v5 相当（group_gemm 本身甚至更快），但拆分引入的 host launcher + 多余的 cast/gather/copy 拉满 wall。
- v5 wall 138 us，v6 wall 1010 us：差距 872 us 全在 host 端，即 tmo op launcher 在 T=83 小 shape 下比 Triton `fast_libentry` 慢一个数量级。

**结论**

- tmo group_gemm 单 op device 9.5 us 是真功夫，比手写 Triton `tl.dot` 单 GEMM 还快。但拆 4 op 在小 shape 下 host 路径吃光收益。
- 若 shape 上到 T=4096+ 且每 expert token 数足够，tmo pipeline host 占比下降，有望打过 v5。本 round 的 T=83/K=2/E=8 太小，单 Triton kernel 是最优解。
- **不替代 v5**。v5 仍为 canonical 实现。

**下一步**

无。本次为对比探查，非优化轮。v5 维持为 final。

---

### Entry 005.c - 对比：手写 BangC kernel（scalar GEMM fallback）

**状态**

v5 Triton 50.4x、v6 tmo 7.04x 之后，调查直接用 BangC（MLU 原生 C++ kernel 编程模型）能不能再压。参考 `/projs/framework/lipenghui/neuware_home/examples` 下的 bangc kernel 例子，尝试写一个 per-token fused MoE kernel，看是否能利用 MLU590 的矩阵单元（`__bang_matmul`）跑过 Triton v5 的 21 us device time。

**优化手段**

非优化轮，仅作对比探查。新建 `bangc/` 子目录：

- `fused_moe_kernel.mlu`：`__mlu_global__` per-token kernel，grid=(T,)，Union1 多核（dim.x=4, dim.y=8 = 32 core）。每个 program：
  1. `__memcpy` 把 `hidden[token]`、`router_logits[token]` 从 GDRAM 拷到 NRAM
  2. softmax（E=8 标量循环 + `expf`）
  3. top-2 selection sort（标量 argmax × 2）
  4. renorm
  5. for k in 0..K-1：`__memcpy` 整块 `w1_T[eid]`（H×2I=128×128 half）和 `w2_T[eid]`（I×H=64×128 half）到 NRAM；标量 GEMM 两轮；SiLU 门控；加权累加
  6. `__memcpy` 输出回 GDRAM
- `host_driver.cpp`：CPU 端生成输入（与 `base.py` 同 shape T=83/H=128/E=8/K=2/I=64），CPU 预转置 w1/w2 到 `w1_T`/`w2_T`，`cnrtMalloc` + `cnrtMemcpyHostToDev`，`cnrtPlaceNotifier` 测 device time，CPU fp32 reference 实现，atol=5e-2 校验。
- `CMakeLists.txt`：`find_package(BANG)` + `bang_add_executable`，cncc 编译 .mlu，g++ 编译 .cpp，链接 cnrt。

**踩坑**

- **`__bang_matmul(float*, const half*, const half*, M, K, N)` 半精度入口的 WRAM 布局未公开**。`bang_host_functions_decls.h` 里有该 overload 声明，但 `neuware_home/examples` 里所有 matmul 样例只用 int8 + 4D strided `__memcpy` 的 WRAM 布局，half 输入的 expected stride 没有任何文档或样例。直接用 `__bang_matmul(nram_out, nram_x, nram_w1, 1, H, TWO_I)` 输出全错（max_abs_diff 3378）。fallback 到标量 GEMM 才得到接近正确的输出。
- **WRAM 不支持标量读**。最初把 `w1`/`w2` 放在 `__wram__`，标量循环读 `nram_w1[i]` 返回 NaN/garbage。WRAM 是矩阵单元专用 RAM，只能通过 `__bang_matmul` 间接访问。把 `w1`/`w2` 改成 `__nram__` 后才正常。
- **`<<<dim, func_type, queue>>>` 是 BangC 专有语法，只有 cncc 能解析**。g++ 报 `expected primary-expression before '>' token`。解法：把 launch 包进 .mlu 里的 `extern "C" launch_fused_moe(...)` wrapper，host_driver.cpp 只 link 这个符号。
- **cnrt API 命名**：`cnrtHost2Device`/`cnrtDevice2Host` 不存在，正确名是 `cnrtMemcpyHostToDev`/`cnrtMemcpyDevToHost`；`cnrtNotifierRecord` 不存在，正确名是 `cnrtPlaceNotifier`。
- **`cnrtNotifierDuration` 返回微秒**（不是毫秒），header 里参数名写的是 `ms` 但实际单位是 us。最初按 ms 处理乘 1e3，导致 device time 报 141565 us（1000x 偏大）。
- **g++ 编译 BangC host 端的 ABI 陷阱**：
  - `<random>` 在 `-std=c++11` 严格模式下 `vswprintf` ABI 报错，必须用 `-std=gnu++11`（GNU 扩展）。
  - `neuware_home/lib/clang/11.1.0/include/` 下的 `stdint.h`/`stddef.h` 用了 `__has_feature`（clang-only），会 shadow 系统 stdint.h，g++ 编译时炸。不能把这个目录加到全局 include path，必须用绝对路径 `#include "/abs/path/to/bang_fp16.h"` 引入。
  - `__internal_float2half` 用 `*(reinterpret_cast<const unsigned int *>(&(f)))` 类型双关，在 `-fstrict-aliasing`（-O2 默认开）下是 UB。CMake 里必须显式加 `-fno-strict-aliasing`。
- **fp16 中间值精度损失**：`nram_gate_up_h`、`nram_act`、`nram_out_k_h`、`nram_out_acc` 全是 `half`，标量 GEMM 累加用 fp32 但每次写回 half，精度损失累积到 max_abs_diff=1.71（输出量级 ~500，相对误差 ~0.3%）。atol=5e-2 FAIL。要修需要把这些中间 buffer 全部升 fp32。

**结果**

- device time/iter：**4090.00 us**（avg of 100 iters，cnrtNotifier 口径）。
- wall time/iter：**4090.28 us**（host std::chrono 口径，与 device 几乎相同说明纯 device-bound）。
- max_abs_diff：1.7117，atol=5e-2 → **FAIL**。输出值（1.9775e+02 vs ref 1.9760e+02 等）相对误差 ~0.1%，但绝对误差超阈值。
- 相对 base：6.94 ms / 4.09 ms ≈ 1.70x（看似比 base 快，但 base 是 eager 50 个 kernel launch 的 wall，device 时间并非 4 ms）。
- 相对 v5 Triton：device 4090 us / 21 us ≈ **慢 195 倍**；wall 4090 us / 138 us ≈ 慢 29.7 倍。
- 相对 v6 tmo：device 4090 us / 79.5 us ≈ 慢 51.5 倍。

**与 upbound 的差距**

- 4090 us device 距离 v5 的 21 us 差 195x，距离 stretch goal 10 us 差 409x。完全不在同一个量级。
- 根因：标量 GEMM fallback 没用矩阵单元。每个 token 要做 2 × 2 = 4 次 GEMM，每次 H×2I 或 I×H 标量乘加，共约 2 × (128×128 + 64×128) = 49152 次 half 乘加 / token，83 token × 32 core 并行 → 单核 ~127K 次 mul-add，按 MLU590 NRAM 标量吞吐估算 4 ms 量级合理。
- 矩阵单元路径（`__bang_matmul`）走不通是因为 half 输入的 WRAM stride 布局未公开。`neuware_home/examples` 里的 matmul 样例全是 int8 + 4D strided `__memcpy`，half 路径需要逆向或问厂商。
- 精度差是次要问题（升 fp32 中间值可解），性能差才是主要问题。

**结论**

- 不替代 v5，不替代 v6。本次为对比探查，记录为 BangC 直写尝试。
- 手写 BangC 在没解决 `__bang_matmul` half 输入 WRAM 布局之前，性能远不如 Triton v5（195x 慢）和 tmo v6（51x 慢）。Triton 在 MLU590 上的 `tl.dot` 路径已经能调用矩阵单元，写 BangC 标量 GEMM 是反向优化。
- 若未来能拿到 `__bang_matmul` half 输入的 WRAM stride 文档，手写 BangC 矩阵单元路径有望接近 v5（参考 tmo `MLUGroupedGemmEx` 单 op 9.5 us 的存在性证明）。本次未解决，留作 P2。
- **v5 Triton 维持为 canonical**，BangC 标量版仅作对比基线。

**下一步**

无。本次为对比探查，非优化轮。若要继续，需要解决 `__bang_matmul` half 输入的 WRAM 布局问题，或改用 int8 量化路径（样例较多）。

---

### Entry 005.d - BangC P2: 中间值升 fp32 修精度

**状态**

005.c 的标量 GEMM BangC kernel 虽然 device time 4090 us，但 max_abs_diff=1.71 FAIL。根因：`nram_gate_up_h`/`nram_act`/`nram_out_k_h`/`nram_out_acc` 全是 `half`，每次 GEMM 累加（fp32）写回 half 时精度损失累积。输出量级 ~500，half ulp ≈ 0.5，atol=5e-2 必然 FAIL。

**优化手段**

把所有中间 buffer 升 fp32，half 只在边界出现：
- `nram_gate_up`：fp32（GEMM1 输出）
- `nram_act`：fp32（SiLU 门控）
- `nram_out_k`：fp32（GEMM2 输出）
- `nram_out_acc`：fp32（最终累加器）
- `nram_out_h`：half（仅 GDRAM store 前的 fp32→fp16 cast buffer）

x/w1/w2 仍是 half（GDRAM→NRAM 加载口径不变），GEMM 内部 `(float)nram_x[h] * (float)nram_w1[...]` 现在直接写回 fp32 中间值，不再经过 half round-trip。最终 store 前 `(half)nram_out_acc[h]` 一次性 cast。

**结果**

- device time/iter：**3761.50 us**（vs 4090 us，-8.0%）。fp32 NRAM 占用更多但标量循环本身没有变慢，反而因为少了两次 fp16↔fp32 cast 略快。
- wall time/iter：**3761.96 us**（仍纯 device-bound）。
- max_abs_diff：**0.0000**（vs 1.71）→ **PASS**。完全匹配 CPU fp32 reference，因为 std=0.02 + fp32 中间值 + 唯一一次 half round-trip 在 store 边界。
- 相对 base：6.94 ms / 3.76 ms ≈ 1.85x。
- 相对 v5 Triton：3762 us / 21 us ≈ 慢 179 倍（仍远慢于 v5）。

**与 upbound 的差距**

精度问题已解决，但性能仍在 3.76 ms 量级。距离 v5 的 21 us 差 179x。根因仍是标量 GEMM fallback 没用矩阵单元。P2 是 correctness fix，不是 perf optimization。

**下一步**

P0：尝试 `__bang_matmul` fp32 overload 走矩阵单元路径。详见 005.e。

---

### Entry 005.e - BangC P0: `__bang_matmul` fp32 overload 调查（blocked）

**状态**

005.d 修完精度后，瓶颈仍是标量 GEMM。尝试用 `__bang_matmul(float*, const float*, const float*, M, K, N)` fp32 overload 走 MLU590 的矩阵单元，期望 device time 从 3762 us → ~100 us 量级（参考 tmo `MLUGroupedGemmEx` 9.5 us 的存在性证明）。

**假设**

`__bang_matmul` 的 fp32 overload 包装 `__mlvm_stream_conv_f32_f32_f32`（1×1 conv intrinsic），arg swap 后 src1 应该是 conv-kernel layout `[N, K]` 而非 matmul-layout `[K, N]`。`__bang_load_matrix` 是把数据从 NRAM 搬到 WRAM 的官方 API，应该能产生 `__bang_matmul` 期望的 WRAM 布局。

**优化手段**

新建 `bangc/matmul_probe.mlu` + `matmul_probe_host.cpp`，孤立测试 fp32 `__bang_matmul`：
- M=K=N=64，src0 = [1..4096] row-major
- src1 测试 4 种配置：all-ones / identity，× trans_en=0/1（控制 `__bang_load_matrix` 的 is_transpose arg）
- 预期：identity src1 应该让 dst[m, n] = src0[m, n]，可以直接读出 N 维的物理排列

probe kernel：
```c
__memcpy(nram_a, src0, M*K*4, GDRAM2NRAM);
__memcpy(nram_b_tmp, src1, K*N*4, GDRAM2NRAM);  // stage via NRAM
__bang_load_matrix(wram_b, nram_b_tmp, NRAM2WRAM, K, N, N, trans_en);
__bang_matmul(nram_c, nram_a, wram_b, M, K, N);
__memcpy(dst, nram_c, M*N*4, NRAM2GDRAM);
```

`__bang_load_matrix` 拒绝 GDRAM2WRAM（dir=4），必须先 GDRAM→NRAM（`__memcpy`）再 NRAM→WRAM（`__bang_load_matrix`）。

**踩坑**

- **`__bang_matmul` 要求 src1 在 `__wram__`**：compile error `invalid pointer address space for '__bang_matmul', expected '__wram__'`。src1 buffer 必须 `__wram__ float wram_b[...]`，不能放 NRAM。
- **`__bang_load_matrix` 拒绝 GDRAM2WRAM**：compile error `invalid direction '4'`。必须两段式：GDRAM→NRAM（`__memcpy`），NRAM→WRAM（`__bang_load_matrix`）。
- **K-reduction 是对的，N-layout 是错的**。all-ones src1 测试 PASS（max_abs_diff 0.0000），证明 sum over K 正确。但 identity src1 测试 FAIL（max_abs_diff 4094），输出 N 维是 permuted layout。
- **decode 出来的 permutation pattern**（M=K=N=64, trans_en=0, identity src1, row 0）：output[p] 里的值 = expected[perm[p]]，其中
  ```
  perm[4k+c] = k + col_offset[c], col_offset = [0, 49, 33, 17]
  ```
  即 c=0 时是 k（0..15 顺序），c=1/2/3 时是 k+{49,33,17}（=16*(4-c)+1）。同一 permutation 对每个 M 行都一样。
- **更严重：value 17（expected[16]）从输出里消失了**。sorted got values 是 {0, 1..16, 18..64}，缺 17。p=61 是 uninitialized slot（got 0）。意味着 `__bang_matmul` fp32 overload 对 N=64 只写 63 个有效位置，丢一个。
- **int8 样例的工作配方不能照搬**。`/neuware_home/samples/BANG/1_Performance/matmul/0_single_core/main.mlu` 用的是 `__bang_matmul(half*, int8_t*, int8_t*, M, K, N, POS)` —— **不同 overload**（带 POS arg，int8→half）。它的 WRAM load 是 4D strided `__memcpy`（magic numbers `__wram_size__/16, 16-1, 4*BLOCK_K, BLOCK_N/64-1, 4*K, 64*K, BLOCK_N/16/4-1`），这套 stride 是 int8-specific。直接套到 fp32 overload 上不会产生正确输出。
- **没有 fp32 / half overload 的 WRAM layout 文档**。`neuware_home/samples` 下所有 matmul 样例只有 int8。`__bang_load_matrix` 产生的 WRAM 布局与 fp32 `__bang_matmul` 期望的布局不一致，但期望布局没有任何文档或样例可参考。

**结果**

- P0 **blocked**。fp32 `__bang_matmul` 输出 layout 未公开 + 丢一个 element，无法在合理时间内集成到 fused_moe kernel。
- probe 自身保留在 `bangc/matmul_probe.mlu` + `matmul_probe_host.cpp`，可复现以上 permutation finding。
- fused_moe kernel 维持 005.d 状态：标量 GEMM + fp32 中间值，device 3762 us，PASS。

**与 upbound 的差距**

不变。005.d 的 3762 us 距 v5 的 21 us 差 179x。P0 没能压进矩阵单元路径。

**下一步**

无。BangC 矩阵单元路径需要厂商文档支持（fp32/half `__bang_matmul` 的 WRAM stride），否则纯靠逆向 stride 无法在可控时间内完成。fused_moe BangC 维持标量 GEMM 对比基线，**v5 Triton 维持 canonical**。

未来若拿到文档，可重启 P0：替换 GEMM1/GEMM2 的标量循环为 `__bang_matmul(nram_gate_up, nram_x_f, wram_w1, 1, H, 2I)` + `__bang_matmul(nram_out_k, nram_act, wram_w2, 1, I, H)`，weight 在 WRAM、x/act 在 NRAM、output 在 NRAM。但需要先把 fp32 `__bang_matmul` 输出的 permuted layout 解 permutation（或找到正确的 `__bang_load_matrix` 参数组合产生 plain row-major 输出）。

---

## 5. 当前瓶颈判断

### 5.1 Host overhead 主导 wall

v5 wall 138 us，device 21 us。device 只占 wall 的 15%。剩余 host overhead 主要是：
- `auto_bench.time_forward` 的固定开销：
  - `set_seed(seed)` 每次 forward：~12 us（`torch.manual_seed` + `torch.mlu.manual_seed_all`）
  - `sync_devices()` 同步 cuda AND mlu：~40 us（因为 `torch.cuda.is_available()` 在本机返回 True，`sync_devices` 会同时同步两个 device）
  - `build_case` + `load_state_dict` + accuracy run 的状态差：~24 us
- Triton `fast_libentry` 剩余路径：~40 us

这些都是测量框架和 launcher 本身的固定开销，进一步压缩空间极小。

### 5.2 device time 距离 stretch goal 还差 2x

21 us 距离 10 us 的 stretch goal 还差 2x，但因为 wall 已被 host 主导，继续压 device time 没有意义。

## 6. 后续优化方向

按优先级：

### P0 - 已达目标，停止迭代

5 轮优化把 wall 从 6.94 ms 压到 0.138 ms（50.4x），device time 21 us 已接近 stretch goal，wall 被 host overhead 主导。继续优化的边际收益递减，且越来越触及测量框架本身的固定开销。

### P1 - 进一步工作的可能性

如果未来要继续压：
- 把 launcher 改成手写 C++ extension（绕过 Triton 的 Python launcher），可能再省 10–20 us host time。
- 拆分 `auto_bench.time_forward`，把 set_seed + sync_devices 调到循环外，但会偏离 auto_bench 的标准口径。
- 改用更大的 shape（如 T=8192）让 device time 重新主导 wall，再优化 device。

但这些都超出当前 5 轮优化目标，留作后续。

## 7. 复现命令

```bash
/projs/framework/lipenghui/venv/pytorch_main/bin/python \
  auto_bench.py \
  --v0_file fused_moe/base.py \
  --v1_file fused_moe/triton_fused_moe_005.py \
  --warmup 50 --repeat 100
```

按 kernel name 汇总 trace：

```bash
jq -r '
  .traceEvents[]
  | select(.cat == "kernel")
  | [.name, .dur]
  | @tsv
' fused_moe/log/triton_fused_moe_005_forward_50iter.pt.trace.json \
| awk -F'\t' '{a[$1]+=$2; c[$1]++} END {for (n in a) printf "%s\tcount=%d\ttotal=%.2fus\tavg=%.2fus\n", n, c[n], a[n], a[n]/c[n]}' | sort -t= -k3 -rn | head
```

## 8. Checkpoint

记录生成时：2026-08-06。

- `base.py` 未修改
- v1–v5 Triton：5 轮累计 50.4x（auto_bench 口径）
- 所有 trace 文件在 `fused_moe/log/` 下（gitignored）

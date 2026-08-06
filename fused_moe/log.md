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

1. 正确性与 wall time 用 `auto_bench.py`，`--warmup 50 --repeat 100`。
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

## 5. 当前瓶颈判断

### 5.1 Host 侧 routing ops 占 wall 主导

v1 device time 只有 21 us，但 wall 564 us。差值主要是：
- PyTorch 的 softmax、topk、renorm、cast 4 个 op 的 launch + sync 开销；
- Python 端组建 `topk_ids`/`topk_weights` tensor 的 host time；
- Triton launcher 自身 host overhead（每调用一次都要 grid/arg 解析）。

## 6. 后续优化方向

按优先级：

### P0 - 把 routing 融进同 kernel

softmax + top-2 + renorm 在 Triton 内部做，省掉所有 routing 相关的 PyTorch op 与 host launch。

### P1 - 削 launcher host overhead

`fast_libentry` + 缓存输出 buffer，把 launcher 路径压到最小。


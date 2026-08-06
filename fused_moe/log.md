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

## 5. 当前瓶颈判断

### 5.1 mask/scatter + Python for-loop 占主导

8 个 expert 串行调度，约 50 个 kernel launch；mask + advanced indexing + scatter 三类 op 占总 device 时间的绝大部分。

## 6. 后续优化方向

按优先级：

### P0 - 写一个 Triton kernel 把 mask/scatter 整体消灭

把 expert 计算整合进单个 kernel，per-token program，循环 top_k 次做 GEMM。

## 7. 复现命令

```bash
/projs/framework/lipenghui/venv/pytorch_main/bin/python \
  auto_bench.py \
  --v0_file fused_moe/base.py \
  --v1_file fused_moe/triton_fused_moe_001.py \
  --warmup 50 --repeat 100
```

按 kernel name 汇总 trace：

```bash
jq -r '
  .traceEvents[]
  | select(.cat == "kernel")
  | [.name, .dur]
  | @tsv
' fused_moe/log/triton_fused_moe_001_forward_50iter.pt.trace.json \
| awk -F'\t' '{a[$1]+=$2; c[$1]++} END {for (n in a) printf "%s\tcount=%d\ttotal=%.2fus\tavg=%.2fus\n", n, c[n], a[n], a[n]/c[n]}' | sort -t= -k3 -rn | head
```

## 8. Checkpoint

记录生成时：2026-08-06。

- `base.py` 未修改
- v1 Triton：待补

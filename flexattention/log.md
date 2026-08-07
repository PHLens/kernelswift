# flexattention Triton Kernel Optimization Log

本文记录 `base.py` 中 scaled_dot_product_attention (SDPA, causal + GQA) 在 MLU590-H8 上的 Triton 优化过程。每次优化独立成 entry，记录当时现状、假设、优化手段、踩坑、结果、与性能上界的差距，以及下一步方向。

## 1. 固定问题与测试口径

### 1.1 算子语义

- 输入：
  - `query: float16[num_tokens, num_heads, head_size]`（83, 8, 64）
  - `key:   float16[num_tokens, num_kv_heads, head_size]`（83, 8, 64）
  - `value: float16[num_tokens, num_kv_heads, head_size]`（83, 8, 64）
- 当前核心 shape：`T=83, H=8, D=64, Kv=8`（GQA ratio = 1）
- 计算等价于：`sdpa(q, k, v, scale=1/sqrt(D), is_causal=True)`，输出 `[T, H*D]`
- 输出：`float16[num_tokens, num_heads * head_size]`（83, 512）

### 1.2 环境

- Device：MLU590-H8
- 可见 MLU core：1（`CUDA_VISIBLE_DEVICES=0`）
- PyTorch：`2.11.0+cpu`
- torch_mlu：`1.32.0+torch2.11.0`
- Triton：`3.2.0`
- Python：`/projs/framework/lipenghui/venv/pytorch_main/bin/python`

### 1.3 测量规则

1. 正确性与 wall time 用 `auto_bench.py`，`--warmup 50 --repeat 100`。所有数据以 auto_bench 为准。
2. device time 以 profiler JSON 中 `cat == "kernel"` 的 `dur` 为准，单位为微秒；单 iter 取 50 次 forward 总和 / 50。
3. wall time 是 `time_forward` 中 `sync_devices` 包裹的中位数。
4. Round 0 baseline 因 `auto_bench` 要求 v1 文件必须含 `ModelNew`，故用 `flexattention/log/measure_base.py` 复刻 `time_forward` 口径（`set_seed` + `sync_devices` per iter，warmup 50 / repeat 100，median），与 auto_bench 同口径。
5. 优化循环每轮选一个明确瓶颈点；不能在同 trace 中稳定改善至少 5% 的方案不进入主实现。

## 2. Upbound 定义

- **工程上界**：单次 fused attention 在 CNNL 层的 reference 实现（类似 `MLUUnion1Tri`，6–10 us / call 量级）。
- **更现实的目标**：把 wall time 压到 ~50 us 量级（单 Triton kernel launch + 少量 host 开销），即可与一次 fused CNNL attention 接近。

## 3. 当前结果总览

| 实现 | Wall time/call (auto_bench) | Kernel device time | 相对上一阶段 | 相对 base |
|---|---:|---:|---:|---:|
| `base.py` eager | 1.9623 ms | ~96.55 us / iter | - | 1.00x |

## 4. Optimization Entries

### Entry 000 - PyTorch eager 起点

**状态**

`base.py` 直接调用 `F.scaled_dot_product_attention(q, k, v, scale=..., is_causal=True)`。前置：`unsqueeze(0)` + `transpose(1, 2)`；GQA ratio = 1 时无 `repeat_interleave`；输出再 `squeeze(0).transpose(0, 1).reshape(T, H*D)`。

50 iter forward 共触发 1100 个 kernel 事件，平均每 forward 22 个 kernel launch，total device work 96.55 us。

**优化手段**

无，记录为基准。

**踩坑**

- `F.scaled_dot_product_attention` 在 MLU 上未走 fused attention kernel，而是分解为 `MLUUnion1StrideBMMGEBB`（QK^T BMM）+ `MLUUnion1KernelSoftmaxForward` + 第二个 BMM（AV），加上 `CastExecOnce` / `SameSizeWhere` / `FillHostValue` / `Transpose` / `IsInfFast` 等小 kernel 共 22 个。
- 最大单 kernel 占比也只有 ~15 us / 14.9 us avg（`MLUBlockKernelExpandB1toBA`），无单点热点。
- `auto_bench.py` 要求 v1 文件必须定义 `ModelNew`，故 round 0 baseline 用 `log/measure_base.py` 复刻 `time_forward` 口径单独测量（同 `set_seed` + `sync_devices` + warmup 50 / repeat 100 / median）。

**结果**

- `measure_base.py` wall（= auto_bench `time_forward` 口径）：`v0=1.9623 ms / call`。
- 50 次 forward 的 kernel device time：96.55 us / iter。
- device_ratio = 96.55 / 1962.3 ≈ 4.9% → **host-bound**（95% wall 在 host 开销）。
- trace：[flexattention_forward_50iter.pt.trace.json](log/flexattention_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- 与 ~10 us 量级的 fused CNNL attention 相差 ~196x；其中 device 只占 5%，主要差距在 22 次 kernel launch + cast / transpose / softmax 等串联的 host 路径。

**下一步**

写一个 Triton kernel 把 SDPA 整体消灭：用 flash-attention 风格的 tiling 在单 kernel 内完成 QK^T / causal mask / softmax / AV，输出直接 `[T, H*D]`，避免 `unsqueeze / transpose / reshape` 触发的额外 cast kernel。

---

## 5. 当前瓶颈判断

### 5.1 host-bound 占绝对主导（device_ratio ≈ 5%）

Base eager 单 forward wall ~1962 us，device 仅 ~97 us / iter（22 个小 kernel 的总和）。`device_ratio = 5%` 落入 **host-bound** 区间，且 22 个 kernel launch 中最大单 kernel 仅 15 us，无单点热点。瓶颈本质是「eager 串行调度 + 多次 cast/transpose/softmax 拆开」，应当用单 Triton kernel 把 SDPA 整段融合掉。

## 6. 后续优化方向

按优先级：

### P0 - v1：单 Triton kernel 实现 fused SDPA（flash-attention tiling）

用 flash-attention 风格的 tiling，一个 grid 里完成 QK^T / scale / causal mask / rowmax / softmax / AV。输出直接按 `[T, H*D]` 行主序写，前置只保留必要的 `reshape`（不引入 cast / transpose）。目标是把 22 个 kernel 压成 1 个，wall 至少降一个数量级到 ~200 us 内。

### P1 - 后续 host 侧（launcher / context / 输出 buffer）

如果 v1 之后 device_ratio 仍 < 20%，按 fused_moe v3/v5 的经验上 `fast_libentry` + 缓存输出 buffer + 移除 `torch.mlu.device()` context，把 launcher host 开销继续往下压。

## 7. 复现命令

```bash
# Round 0 baseline（base.py 无 ModelNew，用 log/measure_base.py 复刻 time_forward 口径）
CUDA_VISIBLE_DEVICES=0 python flexattention/log/measure_base.py

# 后续 round N（v1+）：
/Projs/framework/lipenghui/venv/pytorch_main/bin/python \
  auto_bench.py \
  --v0_file flexattention/base.py \
  --v1_file flexattention/triton_flexattention_<NNN>.py \
  --warmup 50 --repeat 100
```

按 kernel name 汇总 trace：

```bash
jq -r '
  .traceEvents[]
  | select(.cat == "kernel")
  | [.name, .dur]
  | @tsv
' flexattention/log/<file>.pt.trace.json \
| awk -F'\t' '{a[$1]+=$2; c[$1]++} END {for (n in a) printf "%s\tcount=%d\ttotal=%.2fus\tavg=%.2fus\n", n, c[n], a[n], a[n]/c[n]}' | sort -t= -k3 -rn | head
```

## 8. Checkpoint

记录生成时：2026-08-07。

- `base.py` 未修改
- v1–v<NNN> Triton：待补
- 所有 trace 文件在 `flexattention/log/` 下（gitignored）

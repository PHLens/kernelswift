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
4. Round 0 baseline 因 `base.py` 无 `ModelNew`（auto_bench 要求 v1 文件必须含 `ModelNew`），用 `flexattention/log/measure_base.py` 复刻 `time_forward` 口径（`set_seed` + `sync_devices` per iter，warmup 50 / repeat 100，median）；多次运行稳定在 0.96–1.01 ms。
5. 优化循环每轮选一个明确瓶颈点；不能在同 trace 中稳定改善至少 5% 的方案不进入主实现。

## 2. Upbound 定义

- **工程上界**：单次 fused attention 在 CNNL 层的 reference 实现（类似 `MLUUnion1Tri`，6–10 us / call 量级）。
- **更现实的目标**：把 wall time 压到 ~50 us 量级（单 Triton kernel launch + 少量 host 开销），即可与一次 fused CNNL attention 接近。

## 3. 当前结果总览

| 实现 | Wall time/call (auto_bench) | Kernel device time | 相对上一阶段 | 相对 base |
|---|---:|---:|---:|---:|
| `base.py` eager | 1.0060 ms | ~96.69 us / iter | - | 1.00x |
| `triton_flexattention_001.py` v1 | 0.2640 ms | ~96.19 us / iter | 3.81x | 3.81x |
| `triton_flexattention_002.py` v2 | 0.2104 ms | ~50.91 us / iter | 1.26x | 4.56x |

## 4. Optimization Entries

### Entry 000 - PyTorch eager 起点

**状态**

`base.py` 直接调用 `F.scaled_dot_product_attention(q, k, v, scale=..., is_causal=True)`。前置：`unsqueeze(0)` + `transpose(1, 2)`；GQA ratio = 1 时无 `repeat_interleave`；输出再 `squeeze(0).transpose(0, 1).reshape(T, H*D)`。

50 iter forward 共触发 1100 个 kernel 事件，平均每 forward 22 个 kernel launch，total device work 96.69 us。

**优化手段**

无，记录为基准。

**踩坑**

- `F.scaled_dot_product_attention` 在 MLU 上未走 fused attention kernel，而是分解为 `MLUUnion1StrideBMMGEBB`（QK^T BMM）+ `MLUUnion1KernelSoftmaxForward` + 第二个 BMM（AV），加上 `CastExecOnce` / `SameSizeWhere` / `FillHostValue` / `Transpose` / `IsInfFast` 等小 kernel 共 22 个。
- 最大单 kernel 占比也只有 ~15 us（`MLUBlockKernelExpandB1toBA`），无单点热点。
- 首次运行 `measure_base.py` 得到 1.9623 ms（冷启动）；充分预热后稳定在 0.96–1.01 ms，与 auto_bench 的 1.0060 ms 一致。Round 0 表格采用稳态值。
- `auto_bench.py` 要求 v1 文件必须定义 `ModelNew`，故 round 0 baseline 用 `log/measure_base.py` 复刻 `time_forward` 口径单独测量（同 `set_seed` + `sync_devices` + warmup 50 / repeat 100 / median）。

**结果**

- `auto_bench.py` wall（同口径）：`v0=1.0060 ms / call`。
- 50 次 forward 的 kernel device time：96.69 us / iter。
- device_ratio = 96.69 / 1006 ≈ 9.6% → **host-bound**（~90% wall 在 host 开销）。
- trace：[flexattention_forward_50iter.pt.trace.json](log/flexattention_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- 与 ~10 us 量级的 fused CNNL attention 相差 ~100x；其中 device 只占 10%，主要差距在 22 次 kernel launch + cast / transpose / softmax 等串联的 host 路径。

**下一步**

写一个 Triton kernel 把 SDPA 整体消灭：用单 kernel 在一个 program 里完成 QK^T / scale / causal mask / softmax / AV，避免 22 个 launch。

---

### Entry 001 - 单 Triton kernel 实现 fused SDPA

**状态**

Round 0 base eager wall 1.006 ms，device 96.69 us / iter（22 个小 kernel）。`device_ratio = 9.6%` → host-bound，瓶颈在 22 次 launch + cast / transpose / softmax 串行调度，不在 device 本身。

**假设**

- 把 22 个 kernel 合并成 1 个 Triton kernel，wall 应当下降到接近单次 launch + 少量 host 开销（~200–300 us 量级）。
- device time 大致不变（同样 QK^T + softmax + AV 的算术量）。
- 因为 T=83 较小，单 program 单 block softmax 即可，不必上 flash-attention 在线 softmax。

**优化手段**

- 文件：`flexattention/triton_flexattention_001.py`
- kernel `_sdpa_v1_kernel`：grid = `(T, H)`，每个 program 处理一个 (token, head)。一次 load 整个 K/V 块（`T_BLOCK = next_pow2(T) = 128`，causal 行用 mask + other=0.0 跳过），单 pass softmax（`tl.max` + `tl.exp` + `tl.sum`），最后 `tl.store` fp16 输出。
- ModelNew.forward 调用 `_sdpa_v1`，输出直接 `reshape(T, H*D)`，避免 `squeeze / transpose / reshape` 链。GQA ratio > 1 时回退到 `repeat_interleave`（本轮测试 shape 是 ratio=1）。

**踩坑**

- 没有上 `fast_libentry`：v1 目标是把 launch 数从 22 压到 1，launcher 路径本身的优化留给下一轮。
- 用 `tl.where(mask, qk, -float("inf"))` 把被屏蔽位置的 pre-softmax logit 置 `-inf`，softmax 后 `exp(-inf) = 0`，对 `tl.sum` 无副作用。fp32 下 `-inf` 在 MLU 上正常工作。
- `tl.load` 用 `mask=mask[:, None]` + `other=0.0`，避免越界行读到 NaN / 垃圾数据污染 QK^T。

**结果**

- `auto_bench.py` wall：`v0=1.0060 ms, v1=0.2640 ms, speedup=3.811x`，正确性 `PASS`。
- 50 次 forward 的 kernel device time：96.19 us / iter（单 kernel `_sdpa_v1_kernel`，与 base 22 kernel 总 device 96.69 us 基本持平 — 算术量相同）。
- device_ratio = 96.19 / 264 ≈ 36.4% → **mixed**，从 host-bound 移到 mixed 类。
- trace：[triton_flexattention_001_forward_50iter.pt.trace.json](log/triton_flexattention_001_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- device 96.19 us 已接近 base 96.69 us（算术等价），离 ~10 us fused CNNL attention 还差 ~10x device；要继续往下走必须把 device 也压下来（例如 `tl.dot` 替换手写 `tl.sum(q[None,:] * k, axis=1)`，把 GEMM 路径走 BMM 硬件单元而非 elementwise+reduce）。
- wall 264 us 离 ~50 us 目标还差 5x，主要残差在 host launcher 单次开销 ~168 us。

**下一步**

下一轮二选一：(a) 用 `tl.dot` 改写 QK^T 与 AV 把 device time 从 ~96 us 压到 ~30 us 量级（device-bound 优化）；(b) 上 `fast_libentry` + 缓存输出 buffer 把 host launcher 开销从 168 us 往下压。先选 (a)：device_ratio 36% 已不算 host-bound 极限，device 是更大绝对值；如果 (a) 之后 device_ratio 仍 < 20%，再切到 host 路径。

---

### Entry 002 - `tl.dot` 改写 QK^T / AV

**状态**

v1 之后 wall 264 us，device 96.19 us / iter（`_sdpa_v1_kernel` 单 kernel），device_ratio 36.4% → mixed。device 端用 `tl.sum(q[None, :] * k, axis=1)` 实现 QK^T、`tl.sum(p[:, None] * v, axis=0)` 实现 AV，走的是 elementwise + reduce 路径，没走 BMM 硬件单元。

**假设**

- 把 QK^T 改成 `tl.dot(q_2d, tl.trans(k))`（`[1, D] @ [D, T_BLOCK] = [1, T_BLOCK]`），AV 改成 `tl.dot(p_2d, v)`（`[1, T_BLOCK] @ [T_BLOCK, D] = [1, D]`），应该让 device time 显著下降（BMM 硬件单元比 elementwise + reduce 高效）。
- host 路径不变（还是单 launch），wall 下降幅度 ≈ device 下降幅度。

**优化手段**

- 文件：`flexattention/triton_flexattention_002.py`
- `_sdpa_v2_kernel`：把 v1 的 elementwise+reduce 替换成 `tl.dot`。Q 重排为 `[1, D]` fp32；K 转置为 `[D, T_BLOCK]` fp32；`qk_2d = tl.dot(q_2d, k_T)`，reshape 回 `[T_BLOCK]`；softmax 同 v1；P 重排为 `[1, T_BLOCK]`，`out_2d = tl.dot(p_2d, v_f32)`，reshape 回 `[D]`。
- 其余结构（grid=(T,H)、`T_BLOCK=next_pow2(T)`、causal mask、单 pass softmax、输出 reshape）与 v1 一致。

**踩坑**

- `tl.dot` 在 MLU 上要求 2D 输入且 inner dim 匹配。`tl.trans(k)` 把 `[T_BLOCK, D]` 转成 `[D, T_BLOCK]`，inner dim `D` 与 `q_2d` 的 `D` 匹配。
- 输入 dtype：v1 已经在 fp32 下做 elementwise，v2 同样把 k/v cast 到 fp32 喂 `tl.dot`，避免 fp16 BMM 路径可能的精度差异；正确性 max abs diff 0.00195（fp16 epsilon 量级，相对误差 ~0.8%）。
- `tl.reshape(qk_2d, (T_BLOCK,))` / `tl.reshape(out_2d, (D,))`：把 `tl.dot` 输出 `[1, N]` 退成 `[N]` 才能接 softmax 和 `tl.store`。

**结果**

- `auto_bench.py` wall：`v0=0.9604 ms, v1=0.2104 ms, speedup=4.565x`（相对 base 4.56x，相对 v1 1.26x），正确性 `PASS`。
- 50 次 forward 的 kernel device time：50.91 us / iter（单 kernel `_sdpa_v2_kernel`，从 v1 的 96.19 us 降 47%）。
- device_ratio = 50.91 / 210 ≈ 24.2% → 仍在 mixed 区间但向 host-bound 倾斜。
- host 开销 ~159 us / call，与 v1 的 ~168 us 基本持平（说明本轮只动了 device，host 路径未变）。
- trace：[triton_flexattention_002_forward_50iter.pt.trace.json](log/triton_flexattention_002_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- device 50.91 us 仍高于 ~10 us fused CNNL attention 量级，但 `tl.dot` 已经把 elementwise + reduce 替换掉，进一步 device 压缩空间有限（softmax + load/store 仍是必要算术）。
- wall 210 us 离 ~50 us 目标还差 4x，主要残差在 host launcher 单次开销 ~159 us。

**下一步**

下一轮切换到 host 路径：上 `fast_libentry`（class-body `globals()` trick 绕过 `_filter_module_ast`）+ 在 ModelNew 上缓存输出 `torch.empty` buffer，把单次 launcher 开销从 ~159 us 往下压。device 端先放着，等 host 压到接近 floor 再看 device_ratio。

---

## 5. 当前瓶颈判断

### 5.1 mixed：device 51 us / iter + host launcher 159 us / call

v2 之后 wall 210 us，device 51 us（24%），host 159 us（76%）。device_ratio 已从 v1 的 36% 降到 24%，仍在 mixed 区间但 host launcher 已是绝对值最大的非 device 残差。device 端 `_sdpa_v2_kernel` 已走 `tl.dot`，进一步压缩空间有限（softmax + load/store 必要算术）。下一步重点应放在 host launcher 上。

## 6. 后续优化方向

按优先级：

### P0 - v3：`fast_libentry` + 缓存输出 buffer

按 fused_moe v3 经验：`fast_libentry()` 把单次 launcher 从默认 ~60 us 压到 ~10 us 量级；在 ModelNew 上缓存 `torch.empty` 输出避免每次 forward 走 allocator。预期 wall 从 210 us 压到 ~100 us 量级。`fast_libentry` 需在 ModelNew 类体内用 `globals()` trick 绕过 `_filter_module_ast`。

### P1 - v4：移除 `torch.mlu.device()` context

`fast_libentry` 之后，剩余 host 开销里 `torch.mlu.device()` context manager 的 enter/exit 仍是可见项（fused_moe v5 经验：可再降 ~15 us）。直接调用 kernel 不包 context。

### P2 - v5+：`num_warps` / `num_stages` 调参 / flash-attention 在线 softmax

device 端的进一步优化；如果 v3+v4 之后 device_ratio > 50%，可以再上在线 softmax 减少 V 重复 load，或调 `num_warps=2/4` 看 BMM 路径吞吐。

## 7. 复现命令

```bash
# Round 0 baseline（base.py 无 ModelNew，用 log/measure_base.py 复刻 time_forward 口径）
CUDA_VISIBLE_DEVICES=0 python flexattention/log/measure_base.py

# Round N (v1+):
/projs/framework/lipenghui/venv/pytorch_main/bin/python \
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
- v1–v2 Triton：2 轮累计 4.56x（auto_bench 口径）
- 所有 trace 文件在 `flexattention/log/` 下（gitignored）

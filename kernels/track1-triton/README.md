# 赛道一：算子 × 后端 进度矩阵

度量口径：`auto_bench.py`（`--warmup 50 --repeat 100`，median wall）。
表格单元格式：状态 · 加速比 · 最优轮次 · campaign baseline→最佳 candidate wall ms。加速比为累计值；若最佳轮次相对上一 accepted candidate 测量，则使用对应 baseline 与最佳 candidate 的现有 artifacts 计算。
task 编号 ↔ 算子目录映射见 [docs/competition/track1-triton.md](../../docs/competition/track1-triton.md)。

各算子后端的详细结论见 `<算子>/<后端>/final_summary.md`（或 `outcome.md`）；跨后端横向总结见 [summary_all_backends.md](summary_all_backends.md)。

| 算子 | `mlu`（寒武纪 MLU590） | `s60`（燧原 GCU） | `maca`（沐曦 C500） | `bi150`（天数智芯） | `ascend910b`（昇腾） |
|---|---|---|---|---|---|
| `groupedtopk` | ✅ **6.56x** · v4 · **0.840→0.128 ms** | ✅ **1.68x** · r003 · **0.459→0.274 ms** | ✅ **3.29x** · r001 · **0.225→0.068 ms** | ✅ **2.41x** · e2r004 · **0.4835→0.1969 ms** | ✅ **2.84x** · r002 · **0.760→0.267 ms** |
| `flexattention` | ✅ **7.08x** · v3 · **1.006→0.140 ms** | 🟡 **0.94x** · e2r001 · **0.251→0.267 ms**（causal fp16 `tl.dot` 单 kernel，epoch-1 0.42x → 2.2x） | — | 🟡 **1.00x** · e2r003 · **0.150→0.149 ms** | ✅ **1.45x** · r002 · **0.409→0.282 ms** |
| `fused_moe` | ✅ **50.4x** · v5 · **6.940→0.138 ms** | ✅ **13.1x** · r002 · **5.112→0.390 ms**（逐-token 路由 + selection 融合） | — | ✅ **14.81x** · e2r001 · **3.193→0.220 ms** | ✅ **19.4x** · r002 · **7.159→0.369 ms** |
| `sparse_pooler` | ✅ **1.60x** · v4 · **0.910→0.567 ms** | 🟡 **0.79x** · r001 · **0.861→1.092 ms**（epoch-2 确认 measurement-bound：GEMM 61% 厂商库 + 手写 segment-max 慢 4x） | — | ✅ **1.22x** · r001 · **1.070→0.880 ms** | ✅ **1.51x** · r001 · **0.935→0.619 ms** |
| `music_flamingo_rotary_embedding` | 📦 — · — · — | ✅ **1.11x** · e2r001 · **0.449→0.406 ms**（部分融合：freqs 进 kernel，cos/sin 保留 vendor） | ✅ **2.38x** · r001 · **0.191→0.080 ms** | ✅ **1.95x** · r001 · **0.353→0.176 ms** | ✅ **1.86x** · r001 · **0.622→0.334 ms** |
| `mm_encoder_attention` | 📦 — · — · — | 🟡 **0.92x** · e2r002 · **0.2516→0.2750 ms**（fp16 `tl.dot` 单 kernel MHA，epoch-1 0.27x → 3.4x，device-bound 未超 GCU 厂商库） | 🟡 **0.91x** · r002 · **0.116→0.128 ms**（手写 Triton MHA，flash-attn 已最优） | ✅ **1.05x** · e2r003 · **0.1499→0.1423 ms** | 🟡 **0.92x** · r001 · **0.349→0.340 ms** |
| `mhc_post_layer_mix` | 📦 — · — · — | 🟡 **0.77x** · e2r001 · **4.23→5.50 ms**（BLOCK_H 1024 + bf16 registers，epoch-1 0.56x → +37%） | ✅ **31.66x** · r001 · **7.636→0.241 ms** | ✅ **1.20x** · r001 · **8.189→6.427 ms** | ✅ **3.64x** · r001 · **3.198→0.880 ms** |
| `mhc_head_compute_mix` | 📦 — · — · — | ✅ **6.8x** · r001（Sinkhorn 迭代融合） | ✅ **14.07x** · r001 · **1.515→0.118 ms** | ✅ **7.79x** · r001 · **1.433→0.184 ms** | ✅ **9.00x** · r001 · **3.527→0.392 ms** |
| `centre_random_augmentation` | 📦 — · — · — | ✅ **1.90x** · e2r001 · **3.025→1.585 ms**（launch-fusion 96→10，四元数→R+旋转+平移+mask 单 kernel） | — | ✅ **4.49x** · r002 · **1.073→0.239 ms** | ✅ **1.22x** · r001 · **2.463→2.024 ms** |
| `mhc_head_compute_mix_backward` | 📦 — · — · — | ✅ **1.23x** · r001 · **0.40→0.32 ms**（sigmoid-backward 融合，2 小归约 host torch.sum，atomic 不可用） | — | ✅ **1.76x** · r001 · **0.351→0.199 ms** | 🟡 **1.03x** · r001 · **0.446→0.431 ms** |

## 表项说明

- `✅` correctness 通过，已提交 Triton code，wall speedup 达到 `5%` threshold；
- `🟡` correctness 通过，已提交 Triton code，但 wall speedup 未达到 `5%` threshold，仍保留实测加速比；
- `⛔` 没有可接受的 Triton candidate；`—` 表示没有对应 campaign artifact 或没有可测 candidate；
- `📦` 仅有 `base.py`，尚无该后端的 Triton submission。
- 记号 `eMrN`：第 `M` 个 campaign epoch 的第 `N` 轮。历史一轮格子沿用当时的
  `vN`/`rNNN` 记号，未回溯改名。

## 横向对比分析

### 1. `tl.dot` 仍是最大的分水岭

当前五个后端的 `tl.dot` 证据状态可分成四档：

- **campaign-backed**：**MLU**（`fused_moe` / `flexattention` 已用 `tl.dot` 拿到实战收益）
- **probe-backed + 部分 campaign 兑现**：**BI150**（`(32,32)@(32,32)` 的 fp32/bf16 `tl.dot` 已实测，`fused_moe` 已把 per-expert GEMM 融合转成显著 device 收益）
- **已有小规模 probe**：**910B**（`(16,16)@(16,16)` fp32 probe 成功，`num_warps=1/2/4` 也已 probe；当前 attention/GEMM campaign 的主收益来源以 launch/fusion 路径为主，大 shape dot 路线仍待建立）
- **probe-backed（power-of-2 约束）**：**S60**（`tl.dot` 可用但 M/N/K 必须为 2 的幂——`48/80/96/112` 均 FAIL，`16/32/64/128` 通过；`num_warps=1/2/4/8` 已 probe；`mm_encoder_attention` 已用 fp16 `tl.dot` 拿到 0.27x→0.92x 收益）
- **Unknown**：**C500**

因此：

- **C500** 的 attention/GEMM 一旦离不开矩阵单元，通常会退化成 `tl.sum(a*b)` 标量 FMA，并被 mcblas 压制；
- **S60** 的 `tl.dot` 受 2 的幂约束：S=83 只能 pad 到 128（58% FLOP 浪费），attention 撞 GCU 厂商库（TOPS runtime 的 flash-attention）时 device-bound 难翻盘，但相对「无 dot」的标量展开仍有 3~4x 提升；
- **MLU / BI150 / 910B** 的上限更高，分析时需要区分“probe 可用”“候选可编译”“任务 shape 上可兑现收益”三个层级。

### 2. attention 类算子的胜负 = base 库调用开销 × launch 成熟度 × dot 路径是否能兑现

`F.scaled_dot_product_attention` 在所有后端都落到厂商专有 FlashAttention/SDPA，但结局差异极大：

| 后端 | base attention 库 | 小 shape 下库调用开销 | Triton launch 成熟度 | attention 结局 |
|---|---|---|---|---|
| MLU | CNNL SDPA | 高（可被超） | 成熟（fast_libentry） | ✅ 7.08x |
| **910B** | 原生 FA | 偏高 | 成熟 | ✅ 1.45x / 0.92x |
| C500 | flash-attn SDPA | 低 | 一般 | 🟡 0.91x |
| BI150 | Ixmma FA（单 kernel 深度调优） | 极低 | 一般 | 🟡 0.55x/0.61x |
| S60 | GCU 厂商 FA（TOPS runtime） | 低 | 弱 | 🟡 0.92x（mm_encoder）/ 0.94x（flexattention，均 fp16 dot） |

**关键洞察**：910B 的 flexattention 1.45x 说明，attention 胜负同时受当前任务 shape 下的 **base 库调用开销、Triton launch 成熟度、以及可兑现的 dot 路径成熟度** 影响。910B 当前已有小 fp32 `tl.dot` probe，但现有 attention campaign 的主收益来源仍以 launch/fusion 为主；BI150 也已探明 `tl.dot`，不过 base FA（`FlashAttnFwdF16Ixmma`）是单 kernel 且厂商深度调优，13~15 us device 已接近下限，手写 Triton attention 仍难跑赢。

**结论**：attention 胜负由「base 库调用开销 − Triton launch/融合收益」与「任务 shape 下 dot 路径是否成熟」共同决定；GEMM/大矩阵乘类算子的上限则更直接受 `tl.dot` 覆盖面约束。

### 3. 收益主旋律高度一致：kernel fusion（省 launch + 省冗余）

五个后端「赢」的算子，赢法几乎一模一样：

| 算子 | MLU | S60 | C500 | BI150 | 910B |
|---|---|---|---|---|---|
| fused_moe（逐-token 路由融合） | **50.4x** | 13.8x | — | 6.60x | **19.4x** |
| mhc_head_compute_mix（Sinkhorn 20 轮融合） | — | 6.8x | **14.07x** | 7.79x | 9.0x |
| groupedtopk（per-token 融合） | 6.56x | 1.68x | 3.29x | 1.71x⚠️ | 2.84x |

Sinkhorn 融合是跨后端最稳定的大赢家（6.8x~14x）：它是纯 launch-bound 的迭代型算子，谁融合谁赢。逐-token 路由（fused_moe）和 elementwise 链（rotary）同理。

> ⚠️ BI150 的 groupedtopk 走 `torch.compile(mode="reduce-overhead")` 而非 Triton kernel 融合（`torch.topk` tie 语义锁死直接重写），收益来自 host launch 压缩，device 内核未动。

### 4. 两个「超额」单点：MLU 的 launcher、C500 的 tiny-K GEMM

- **MLU** 仍是唯一明确打赢厂商 attention 库的后端（flexattention 7.08x），靠的是 `fast_libentry` 快速 launcher + 已兑现的 `tl.dot` 路线；base 的 CNNL SDPA 在 T=83 下 host 调用开销高，被 Triton 的轻量 launcher 反超。
- **C500 的 mhc_post_layer_mix 31.66x** 是全赛道最大单算子加速：base 的 einsum（K=4）落到 `mcblas tf32gemm 64x64x128` tile，K 维浪费 ~97%，手写 kernel 用 4 次显式 fp32 MAC 替代并折叠 elementwise 尾，6→1 kernels。这是「识别厂商库 tile 浪费」的典型案例。

### 5. 证据质量本身也是后端差异

- **MLU / C500**：device kernel 证据相对直接，Designer/Verifier 更容易判断瓶颈；
- **S60**：当前只有 `gcu_runtime` launch 事件，没有 `cat=kernel` device-duration，很多判断只能停留在 launch 诊断（device 时间靠 wall − launch-API 反推）；已建立 machine-readable `triton_gcu` profile（`tl.dot` power-of-2、`num_warps 1/2/4/8` 两条 approved probe evidence）；
- **910B**：device time 可得，但要走 `torch_npu.profiler` + CANN sqlite，不能把 raw `torch.profiler` 当成 device 证据；
- **BI150**：campaign 已有 device 侧总结，但 target profile 对 profiler 字段仍偏保守，后续仍应补正式 probe / promote。

这意味着 backend campaign 的难度，不只取决于 Triton 能不能写，还取决于 **能不能稳定观察 device bottleneck**。

## 后端能力矩阵

| 维度 | MLU590 | S60 | C500 | BI150 | 910B |
|---|---|---|---|---|---|
| `tl.dot` | ✅ campaign-backed | ⚠️ probe-backed：可用但 M/N/K 须为 **2 的幂**（96=16×6 FAIL），fp16/fp32/bf16 均正确 | ❌ Unknown | ✅ probe-backed + `fused_moe` 已兑现 | ⚠️ `(16,16)` fp32 probe-backed，任务 shape 仍待验证 |
| `num_warps>1` | ⚠️ `2` 已失败，当前 `1` 最稳 | ✅ `1/2/4/8` 已 probe（fp16 dot 下 `1` 最优） | ❌ 未建立 | ⚠️ Unknown | ✅ `1/2/4` 已 probe |
| 快速 launch 机制 | ✅ `fast_libentry` | — | — | ⚠️ direct launch + `torch.compile(reduce-overhead)`，无已证明 fast launcher | ✅ 成熟 launch |
| 设备侧 profiler 证据 | ✅ 相对成熟 | ❌ launch-only | ✅ 有 kernel events | ⚠️ campaign 有 summary，profile 仍待补齐 | ✅ 经 CANN/msprof 可得 |
| 厂商库压制力 | 中（attention 可被超） | 强（GCU 厂商库） | 强（mcblas） | 强（Ixmma/TCU） | 强（原生 FA） |
| 覆盖完整度 | 4/10 | 10/10 | 5/10 | **10/10** | 10/10 |

## 结论

1. **当前已有四类 `tl.dot` 证据**：MLU 属于 campaign-backed，BI150 属于 probe-backed 且已有 `fused_moe` 实战收益，910B 已有小 fp32 probe，S60 属于 probe-backed（power-of-2 约束）且已在 `mm_encoder_attention` 兑现 0.27x→0.92x；仅 C500 仍把 `tl.dot` 可用性列为第一优先级。

2. **赛道主旋律仍是「kernel fusion 对抗厂商库」**：能绕开 GEMM/attention 主核的算子（Sinkhorn、逐-token 路由、elementwise 链），Triton 融合普遍拿到 6x~50x；撞上厂商张量核心主路径的算子（大 GEMM、flash attention），收益取决于 dot 路线成熟度和 baseline 库的接近下限程度。

3. **`tl.dot` 仍是全局杠杆点**：分析时需要区分三个层级——`probe 可用`、`候选可编译`、`任务 shape 上可兑现收益`。当前 C500 缺第一层；S60 已补第一层（power-of-2 约束）并在 attention 上兑现第二层；910B / BI150 还在补第三层，MLU 已证明部分路径。

4. **两条独立的胜负线**：GEMM 类看矩阵单元能力与 `tl.dot` 覆盖面；attention 类还要看「base 库调用开销 vs Triton launch/融合收益」。910B 当前的 flexattention 1.45x 主要体现后者，现有证据对应的小规模 `tl.dot` probe 仍不足以外推出全面替代厂商 FA 的结论。

5. **可观察性也是 backend 能力的一部分**：S60 当前的主要短板是 device-duration 证据缺失；910B 和 BI150 则仍有 profile、summary、campaign 三层证据同步吸收的工程工作量。

## 各后端详细总结

各后端（含 MLU590 / S60 / C500 / BI150 / Ascend 910B）的算子级明细、根因分析、可优化方向见 **[summary_all_backends.md](summary_all_backends.md)**。

## 维护

新增/完结 campaign 时更新进度矩阵（目前手动维护，`scripts/update_matrix.py` 自动汇总尚未实现）。

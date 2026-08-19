# 赛道一：算子 × 后端 进度矩阵

度量口径：`auto_bench.py`（`--warmup 50 --repeat 100`，median wall）。
表格单元格式：状态 · 加速比 · 最优轮次 · campaign baseline→最佳 candidate wall ms。加速比为累计值；若最佳轮次相对上一 accepted candidate 测量，则使用对应 baseline 与最佳 candidate 的现有 artifacts 计算。
task 编号 ↔ 算子目录映射见 [docs/competition/track1-triton.md](../../docs/competition/track1-triton.md)。

各算子后端的详细结论见 `<算子>/<后端>/final_summary.md`（或 `outcome.md`）；跨后端横向总结见 [summary_all_backends.md](summary_all_backends.md)。

| 算子 | `mlu`（寒武纪 MLU590） | `s60`（燧原 GCU） | `maca`（沐曦 C500） | `bi150`（天数智芯） | `ascend910b`（昇腾） |
|---|---|---|---|---|---|
| `groupedtopk` | ✅ **6.56x** · v4 · **0.840→0.128 ms** | ✅ **1.68x** · r003 · **0.459→0.274 ms** | ✅ **3.29x** · r001 · **0.225→0.068 ms** | ✅ **1.71x** · r009 · **0.475→0.277 ms** | ✅ **2.84x** · r002 · **0.760→0.267 ms** |
| `flexattention` | ✅ **7.08x** · v3 · **1.006→0.140 ms** | 🟡 **0.42x** · r001 · **0.269→0.64 ms**（correctness PASS，手写 causal SDPA，慢因 `tl.dot` 缺失） | — | 🟡 **0.61x** · r001 · **0.150→0.238 ms** | ✅ **1.45x** · r002 · **0.409→0.282 ms** |
| `fused_moe` | ✅ **50.4x** · v5 · **6.940→0.138 ms** | ✅ **13.1x** · r002 · **5.112→0.390 ms**（逐-token 路由 + selection 融合） | — | ✅ **6.60x** · r002 · **3.259→0.493 ms** | ✅ **19.4x** · r002 · **7.159→0.369 ms** |
| `sparse_pooler` | ✅ **1.60x** · v4 · **0.910→0.567 ms** | 🟡 **0.79x** · r001 · **0.861→1.092 ms** | — | ✅ **1.22x** · r001 · **1.070→0.880 ms** | ✅ **1.51x** · r001 · **0.935→0.619 ms** |
| `music_flamingo_rotary_embedding` | 📦 — · — · — | 🟡 **0.9x** · r002（elementwise 融合，measurement-bound） | ✅ **2.38x** · r001 · **0.191→0.080 ms** | ✅ **1.95x** · r001 · **0.353→0.176 ms** | ✅ **1.86x** · r001 · **0.622→0.334 ms** |
| `mm_encoder_attention` | 📦 — · — · — | 🟡 **0.27x** · r001（手写 SDPA，慢因 `tl.dot` 缺失） | 🟡 **0.91x** · r002 · **0.116→0.128 ms**（手写 Triton MHA，flash-attn 已最优） | 🟡 **0.55x** · r001 · **0.151→0.239 ms** | 🟡 **0.92x** · r001 · **0.349→0.340 ms** |
| `mhc_post_layer_mix` | 📦 — · — · — | 🟡 **0.56x** · r001（einsum 用 `tl.sum` 展开） | ✅ **31.66x** · r001 · **7.636→0.241 ms** | ✅ **1.20x** · r001 · **8.189→6.427 ms** | ✅ **3.64x** · r001 · **3.198→0.880 ms** |
| `mhc_head_compute_mix` | 📦 — · — · — | ✅ **6.8x** · r001（Sinkhorn 迭代融合） | ✅ **14.07x** · r001 · **1.515→0.118 ms** | ✅ **7.79x** · r001 · **1.433→0.184 ms** | ✅ **9.00x** · r001 · **3.527→0.392 ms** |
| `centre_random_augmentation` | 📦 — · — · — | 🟡 **0.95x** · r001（四元数旋转） | — | ✅ **4.49x** · r002 · **1.073→0.239 ms** | ✅ **1.22x** · r001 · **2.463→2.024 ms** |
| `mhc_head_compute_mix_backward` | 📦 — · — · — | 🟡 **1.26x** · r001（sigmoid-backward 融合） | — | ✅ **1.76x** · r001 · **0.351→0.199 ms** | 🟡 **1.03x** · r001 · **0.446→0.431 ms** |

## 表项说明

- `✅` correctness 通过，已提交 Triton code，wall speedup 达到 `5%` threshold；
- `🟡` correctness 通过，已提交 Triton code，但 wall speedup 未达到 `5%` threshold，仍保留实测加速比；
- `⛔` 没有可接受的 Triton candidate；`—` 表示没有对应 campaign artifact 或没有可测 candidate；
- `📦` 仅有 `base.py`，尚无该后端的 Triton submission。

## 横向对比分析

### 1. `tl.dot` 可用性是最大的分水岭（GEMM 类算子）

五个后端里，**只有 BI150 实测 `tl.dot` 可用**（fp32/bf16 精确、fp16 收缩 128/64 可用、M≥16 warp-tile 约束）。C500 / S60 / 910B 三个后端的 `tl.dot` 都是 Unknown，任何 GEMM 只能退化成 `tl.sum(a*b)` 标量 FMA：

- **S60** 因此 attention 跌到 0.27x/0.42x、post_layer_mix 0.56x（标量 FMA vs CNNL 张量核心，差一个数量级）
- **C500** 的 mm_encoder_attention 只 0.91x（同因）
- **BI150** 的 fused_moe 能用 `tl.dot` 融合 per-expert GEMM，拿到 +79.98% device 收益——这是其它 `tl.dot`-Unknown 后端做不到的

### 2. attention 类算子的胜负 = base 库调用开销 vs Triton launch 成熟度

`F.scaled_dot_product_attention` 在所有后端都落到厂商专有 FlashAttention/SDPA，但结局差异极大：

| 后端 | base attention 库 | 小 shape 下库调用开销 | Triton launch 成熟度 | attention 结局 |
|---|---|---|---|---|
| MLU | CNNL SDPA | 高（可被超） | 成熟（fast_libentry） | ✅ 7.08x |
| **910B** | 原生 FA | 偏高 | 成熟 | ✅ 1.45x / 0.92x |
| C500 | flash-attn SDPA | 低 | 一般 | 🟡 0.91x |
| BI150 | Ixmma FA（单 kernel 深度调优） | 极低 | 一般 | 🟡 0.55x/0.61x |
| S60 | CNNL FA | 低 | 弱 | ❌ 0.27x/0.42x |

**关键洞察**：910B 在 `tl.dot` 缺失的情况下，flexattention 仍拿到 1.45x——因为 base 的原生 FA 在 T=83 小 shape 下没吃饱、launch 开销占比高，而 910B 的 Triton launch 成熟，靠「减少 launch + 融合 causal mask/softmax」反而跑赢。BI150 没赢则是因为 base FA（`FlashAttnFwdF16Ixmma`）是单 kernel 且厂商深度调优，13~15 us device 已接近下限。

**结论**：attention 胜负由「base 库调用开销 − Triton launch 开销」的差值决定，而非单纯 `tl.dot`；GEMM/大矩阵乘类算子的上限才真正由 `tl.dot` 决定。

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

- **MLU** 是唯一能「打赢厂商 attention 库」的后端（flexattention 7.08x），靠的是 `fast_libentry` 快速 launcher——base 的 CNNL SDPA 在 T=83 下 host 调用开销高，被 Triton 的轻量 launcher 反超。
- **C500 的 mhc_post_layer_mix 31.66x** 是全赛道最大单算子加速：base 的 einsum（K=4）落到 `mcblas tf32gemm 64x64x128` tile，K 维浪费 ~97%，手写 kernel 用 4 次显式 fp32 MAC 替代并折叠 elementwise 尾，6→1 kernels。这是「识别厂商库 tile 浪费」的典型案例。

## 后端能力矩阵

| 维度 | MLU590 | S60 | C500 | BI150 | 910B |
|---|---|---|---|---|---|
| `tl.dot` | ✅ 可用 | ❌ Unknown | ❌ Unknown | ✅ Supported | ❌ Unknown |
| `num_warps>1` | ⚠️ 不支持（回退 1） | ❌ | ❌ | ⚠️ Unknown | ⚠️ Unknown |
| 快速 launch 机制 | ✅ `fast_libentry` | — | — | ⚠️（CUDA Graph） | ✅ 成熟 launch |
| 厂商库压制力 | 中（attention 可被超） | 强（CNNL） | 强（mcblas） | 强（Ixmma/TCU） | 强（原生 FA） |
| 覆盖完整度 | 4/10 | 10/10 | 5/10 | **10/10** | 10/10 |

## 结论

1. **BI150 是唯一 `tl.dot` 可用的后端**，这是它相对其它后端最独特的优势——虽因厂商库挡路未在 attention/大 GEMM 上兑现，但 fused_moe 的 +79.98% device 收益就是 `tl.dot` 的直接红利。

2. **赛道主旋律是「kernel fusion 对抗厂商库」**：能绕开 GEMM/attention 的算子（Sinkhorn、逐-token 路由、elementwise 链），Triton 融合普遍拿到 6x~50x；撞上厂商张量核心的算子（大 GEMM、flash attention），在 `tl.dot` 不可用或库深度调优的后端上回退。

3. **`tl.dot` 是下一个全局杠杆点**：C500/S60/910B 不约而同把「实测 `tl.dot` 可用性」列为第一优先级。一旦补上，attention 和 post_layer_mix 的 0.27x~0.92x 都有翻盘空间，BI150 已经证明这条路是通的。

4. **两条独立的胜负线**：GEMM 类看 `tl.dot`，attention 类看「base 库调用开销 vs Triton launch 成熟度」。910B 证明了后者可以在 `tl.dot` 缺失时仍赢 attention（flexattention 1.45x）。

## 各后端详细总结

各后端（含 MLU590 / S60 / C500 / BI150 / Ascend 910B）的算子级明细、根因分析、可优化方向见 **[summary_all_backends.md](summary_all_backends.md)**。

## 维护

新增/完结 campaign 时更新进度矩阵（目前手动维护，`scripts/update_matrix.py` 自动汇总尚未实现）。

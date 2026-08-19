# 赛道一：各后端 Triton 优化详细总结

本文是 5 个硬件后端（MLU590 / S60 / C500 / BI150 / Ascend 910B）的算子级详细总结，含根因分析与可优化方向。总览进度矩阵与横向对比见 [README.md](README.md)。

度量口径统一为 `auto_bench.py --warmup 50 --repeat 100`（median wall time），speedup = v0(base.py) / v1(candidate)。

---

## 一、MLU590（寒武纪）

4 个算子 campaign，见各算子 `mlu/outcome.md`。

| 算子 | 提交物 | speedup | 手段 |
|---|---|---|---|
| `groupedtopk` | `triton_grouped_topk_004.py` | **6.56x**（0.840→0.128 ms） | 单 per-token kernel 融合 + `tl.argmax` selection sort + host 端 constexpr 预计算 |
| `flexattention` | `triton_flexattention_003.py` | **7.08x**（1.006→0.140 ms） | QK^T/softmax/AV 用 `tl.dot` + `fast_libentry` + 输出缓存，22→1 launch |
| `fused_moe` | `triton_fused_moe_005.py` | **50.4x**（6.940→0.138 ms） | 单 kernel 融合 softmax+topk+per-expert GEMM+SiLU+加权归约，`tl.dot` 走 BMM 硬件单元 |
| `sparse_pooler` | `triton_sparse_pooler_004.py` | **1.60x**（0.910→0.567 ms） | host allocation reuse（见 `mlu/rounds/`） |

### 关键洞察

MLU 是唯一「打赢厂商 attention 库」的后端（flexattention 7.08x），关键在 **`fast_libentry` 快速 launcher + `tl.dot` 可用** 两者叠加——base 的 CNNL SDPA 在 T=83 小 shape 下 host 调用开销高，被 Triton 的轻量 launcher 反超。

### 独有经验（可复用）

- **`fast_libentry` 调用形式**：是 `fast_libentry()(_kernel)`（工厂调用），不是 `fast_libentry(_kernel)`，写错会静默不生效。
- **`tl.argmax` selection sort**：Triton 无内置 top-K，靠「找 max → mask → 重复 K 趟」实现，groupedtopk 的 K=8+K=4 共 12 趟 reduction 是 device 大头。
- **`torch.cuda.is_available()` 在 MLU 机返回 True**：`sync_devices` 会同步 cuda stub + mlu 两边，每次 forward 多 ~66 us，属 harness 固定成本。
- **tmo 库 op 上界对比**：`moe_softmax_topk` / `flash_attention` 单 op 在 device 上优于手写（groupedtopk 9.86 vs 20 us、flexattention 8.77 vs 50.5 us），但 wall 上不一定（tmo launcher 在小 shape 下重一个数量级），仅作工程上界参考，不作 canonical。

---

## 二、S60（燧原 GCU）

10/10 算子全部 correctness PASS，无留空。

| 算子 | speedup | 手段 |
|---|---|---|
| `fused_moe` | **13.8x** | 逐-token 路由 + selection 融合，省 launch 与冗余计算 |
| `mhc_head_compute_mix` | **6.8x** | Sinkhorn 20 轮迭代融合进单 kernel |
| `groupedtopk` | **1.68x** | 12→1 launch 融合 + 输出池复用 |
| `mhc_head_compute_mix_backward` | **1.26x** | elementwise sigmoid-backward 融合 |
| `centre_random_augmentation` | 0.95x | 四元数旋转（随机数 host 生成） |
| `music_flamingo_rotary_embedding` | 0.9x | 纯 elementwise 融合，measurement-bound |
| `mhc_post_layer_mix` | 0.56x | einsum 用 tl.sum 展开 |
| `flexattention` | 0.42x | 手写 causal SDPA |
| `mm_encoder_attention` | 0.27x | 手写 SDPA |
| `sparse_pooler` | — | 库算子占优，正确性优先提交 |

### 根因：`tl.dot` Unknown

GCU 的 `triton_gcu` 未实现 `tl.dot`，任何 GEMM 退化成 `tl.sum(a*b)` 标量 FMA，无法利用 Matrix Core。手写 SDPA 三处致命退化：

1. QK^T 与 PV 用标量乘加（未用张量核心）
2. `num_warps=1` 且每 query 一个 program，K/V 无跨 program 复用
3. fp16 输入全程转 fp32，失去 fp16 张量核心吞吐

而 base 的 SDPA/einsum 都 dispatch 到 CNNL（汇编级张量核心 + fp16 优化），**Triton 标量 FMA vs CNNL 张量核心，天然差一个数量级**。

### 可优化方向

1. 提高 `num_warps` 与 tile 并行度（预计 0.27x→0.5x，但无法追平张量核心）
2. 实测 `tl.dot` 可用性（哪怕降级实现也比纯 `tl.sum` 好）
3. 混合方案：核心 GEMM 用 `torch.mm`（CNNL），Triton 只做 softmax/mask 融合

---

## 三、C500（沐曦 MACA）

5/10 算子 correctness PASS（`flexattention`、`fused_moe`、`sparse_pooler`、`centre_random_augmentation`、`mhc_head_compute_mix_backward` 暂无 maca 提交物）。

| 算子 | speedup | 手段 |
|---|---|---|
| `mhc_post_layer_mix` | **31.66x** | tiny-K GEMM 融合（K=4 显式 fp32 MAC），6→1 kernels |
| `mhc_head_compute_mix` | **14.07x** | sigmoid gates + 20 轮 Sinkhorn 迭代融合，133→1 launches |
| `groupedtopk` | **3.29x** | 单 per-token kernel 融合 softmax+top4+top8+重归一化 |
| `music_flamingo_rotary_embedding` | **2.38x** | 11→1 elementwise 融合 |
| `mm_encoder_attention` | 0.91x | 手写 Triton MHA |

### 亮点：mhc_post_layer_mix 31.66x（全赛道最大单算子加速）

base 的 einsum `'abmn,abmc->abnc'`（K=4）落到 `mcblas tf32gemm 64x64x128` tile，K 维浪费 ~97%（6071 us/call = 80% device）；手写 kernel 用 4 次显式 fp32 MAC 替代并折叠 elementwise 尾（mul+add+2 bf16 cast），6→1 kernels。r002 后到 **memory-bandwidth floor**（~1 TB/s，接近 C500 HBM 上限）。

### 根因：`tl.dot` / `num_warps>1` 均 Unknown

- `mm_encoder_attention` 0.91x：QK^T/PV 用 `tl.sum(q*K)` 标量乘加（head_size=64 手动 dot），剩余主导成本 `_mha_fwd_kernel` ~64.85 us/call 无 `tl.dot` 无法再加速。
- `mhc_head_compute_mix` 到 latency floor（Sinkhorn 串行依赖链不可并行）。

---

## 四、BI150（天数智芯）

10/10 算子全部覆盖，**唯一 `tl.dot` 可用的后端**。

### 1. 成果总表

| 算子 | 优化结果 | 手段 | 关键数字 |
|---|---|---|---|
| groupedtopk | +41.6%（1.71x） | `torch.compile(mode="reduce-overhead")`（非 Triton） | wall 0.475→0.277 ms |
| flexattention | 无优化空间 | naive Triton causal attention 交付物（正确，0.612x） | base 为厂商 Ixmma FlashAttention |
| fused_moe | +84.9%（6.60x） | per-expert dispatch 融合 + tl.dot GEMM 融合 | 123.9→9.82 kernels/call |
| sparse_pooler | +17.0%（1.22x） | log1p(relu) + per-sequence max-pooling 融合 | 11.92→6.88 kernels/call |
| music_flamingo_rotary_embedding | +48.64%（1.95x） | kernel fusion（13 elementwise → 1） | 13→1 kernels/call |
| mm_encoder_attention | 无优化空间 | naive Triton attention 交付物（正确，0.547x） | base 为厂商 Ixmma FlashAttention |
| mhc_post_layer_mix | +20.09%（1.20x） | elementwise 尾融合（GEMM 不动） | 5.66→2.96 kernels/call |
| mhc_head_compute_mix | +87.17%（7.79x） | Sinkhorn 20 轮迭代 + elementwise 全融合 | 132.88→1.0 kernels/call |
| centre_random_augmentation | +77.7%（4.49x） | 确定性计算全链融合（RNG 留 host） | 78.8→5.52 kernels/call |
| mhc_head_compute_mix_backward | +43.11%（1.76x） | sigmoid backward + 两个 reduce 融合 | 9.74→2.96 kernels/call |

### 2. device time 视角

| 算子 | Device 基线 (us/call) | Device 优化后 | Device 提升 | 手段 |
|---|---|---|---|---|
| mhc_head_compute_mix | 926.40 | 12.996 | −98.6%（71x） | Sinkhorn + elementwise 全融合 |
| fused_moe | 968.16 | 140.84 | −85.5%（6.87x） | dispatch 融合 + tl.dot GEMM 融合 |
| mhc_head_compute_mix_backward | 185.60 | 14.692 | −92.1%（12.6x） | sigmoid backward + reduce 融合 |
| centre_random_augmentation | 420.68 | 29.24 | −93.0%（14.4x） | 确定性计算全链融合 |
| music_flamingo_rotary_embedding | 68.64 | 30.829 | −55.1%（2.2x） | 13 elementwise → 1 |
| mhc_post_layer_mix | 7323.85 | 6122.54 | −16.4%（1.20x） | elementwise 尾融合 |
| sparse_pooler | 743.06 | 609.40 | −18.0%（1.22x） | 激活 + pooling 融合 |

device 提升远大于 wall 提升，正说明这是 kernel 优化成果（融合后 wall 被 harness 固定 host 开销锁死，device_ratio 降到 0.07~0.12）。真正动 kernel 的 7 算子 device 合计约 10.7→7.0 ms。

> ⚠️ groupedtopk 的 device 数据有水分：CUDA Graph 回放使 profiler 无法归因 graph 内部 kernel，`device_us_per_call` 从 109.2「看似」掉到 14.9，但真实 device 里 `gatherTopK`(48.3us)+`bitonicSortKVInPlace`(36.8us) 依然存在（`torch.topk` tie 语义锁死）。该算子 device 基本没动，收益全来自 host launch 压缩。

### 3. 是优化 launch time 吗？

- **kernel fusion（主力，7 算子）**：省 device 侧碎片 kernel 的调度/同步 gap + 中间结果反复写读显存，顺带减少 launch。以 head_compute_mix 为例，~120/133 个 Sinkhorn 小 kernel 每处理 256 元素，融合后数据全程留寄存器，device 926→13 us。
- **torch.compile reduce-overhead（groupedtopk）**：靠 CUDA Graph 压缩 host dispatch/launch 开销，device 内核没动。这才是真正的「优化 launch time」。

### 4. Triton 原语能力矩阵（实测）

| 原语 | 实测结果 |
|---|---|
| `tl.dot` | Supported（fp32/bf16 (32,32) 精确；fp16 收缩 128/64 正确 lower；M≥16 warp-tile 约束） |
| `tl.sqrt/sin/cos` | Supported（与 torch 逐位一致，max_abs_diff=0.0） |
| `tl.sigmoid`/`tl.sum`/`tl.atomic_add` | Supported |
| `tl.static_range` | 小迭代可用，大迭代（19 次）编译爆炸，用动态 `tl.range` 替代 |
| `num_warps`/`num_stages`/block pointers | 仍 Unknown |

### 5. Triton 的边界与本批客观限制

本批成果受限于「小 shape + 厂商库挡路 + BI150 Triton 后端不成熟」：

- **shape 太小**：多数算子输入几百~几千元素，任何 kernel 都被 launch/调度主导，融合掉碎片即达 device 理论下限。
- **厂商库挡路**：attention 的 `FlashAttnFwdF16Ixmma`、GEMM 的 `gemm_tcu_h` 都是 Iluvatar 用 Ixmma/TCU 专门手调，`tl.dot` 无证据能映射到厂商 TCU，故只融合周边 elementwise 尾巴。
- **fused_moe vs sparse_pooler 对比**：同为含 GEMM 算子，结局因瓶颈类型而异——fused_moe launch-bound（123.9 kernel，tl.dot 融合大赢 +79.98%），sparse_pooler compute-bound（2 个 GEMM launch，需算得比 TCU 快，vendor-optimal-bound stop）。

若换成大 shape、compute-bound、无厂商库挡路的算子，Triton 还能做出数量级成绩。

---

## 五、Ascend 910B（昇腾）

10/10 算子全部 correctness PASS，无留空。环境：Ascend910B4、Triton-Ascend 3.2.1、CANN 9.0.0。

| 算子 | speedup | 手段 |
|---|---|---|
| `fused_moe` | **19.4x** | 逐-token 路由 + per-expert GEMM 融合 + host 输出缓存 |
| `mhc_head_compute_mix` | **9.0x** | Sinkhorn 20 轮融合（device 282→8.8 us） |
| `mhc_post_layer_mix` | **3.64x** | einsum 融合 |
| `groupedtopk` | **2.84x** | 单 per-token kernel 融合 + 输出 buffer 复用 |
| `music_flamingo_rotary_embedding` | **1.86x** | 逐-token cos/sin 计算融合 |
| `sparse_pooler` | **1.51x** | log1p+ReLU+max-pool 融合 |
| `flexattention` | **1.45x** | causal SDPA 单 kernel 融合 |
| `centre_random_augmentation` | **1.22x** | 四元数旋转矩阵融合 |
| `mhc_head_compute_mix_backward` | 1.03x | sigmoid-backward 融合（<5% 阈值） |
| `mm_encoder_attention` | 0.92x | 手写 Triton attention |

### 关键洞察

- **`flexattention` 1.45x**：910B 在 `tl.dot` 缺失下仍赢 attention——base 原生 FA 在 T=83 小 shape 下没吃饱、launch 开销占比高，Triton launch 成熟，靠减少 launch + 融合 causal mask/softmax 反超。
- **多数算子 host-bound**：device_ratio 0.07~0.33（groupedtopk 0.131、flexattention 0.19、fused_moe 0.07、sparse_pooler 0.33），wall 主要由 host launch/dispatch 主导。
- **`mm_encoder_attention` 0.92x**：`tl.dot` Unknown，手写 SDPA 无法利用矩阵单元，base 原生 FA 已接近最优。

### 可优化方向

1. 验证 `tl.dot` / `fast_libentry` 在 triton_ascend 的可用性（attention/GEMM 有明确双收益）
2. 压缩 host launch 开销（~107-232 us/call 是小 shape 算子的最大单点）
3. 增大 tile / num_warps 提升 occupancy（但追平库算子仍需矩阵单元）

---

## 六、后端能力矩阵

| 维度 | MLU590 | S60 | C500 | BI150 | 910B |
|---|---|---|---|---|---|
| `tl.dot` | ✅ 可用 | ❌ Unknown | ❌ Unknown | ✅ Supported | ❌ Unknown |
| `num_warps>1` | ⚠️ 不支持（回退 1） | ❌ | ❌ | ⚠️ Unknown | ⚠️ Unknown |
| 快速 launch 机制 | ✅ `fast_libentry` | — | — | ⚠️（CUDA Graph） | ✅ 成熟 launch |
| 厂商库压制力 | 中（attention 可被超） | 强（CNNL） | 强（mcblas） | 强（Ixmma/TCU） | 强（原生 FA） |
| 覆盖完整度 | 4/10 | 10/10 | 5/10 | **10/10** | 10/10 |

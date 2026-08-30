# S60（燧原 GCU）Triton 优化经验沉淀

> 来源：6 个算子的 S60 epoch2 campaign（mm_encoder_attention、flexattention、
> mhc_post_layer_mix、centre_random_augmentation、music_flamingo_rotary_embedding、
> sparse_pooler）。
> 本文沉淀的是**可跨算子复用**的后端能力事实与判断方法论，不是该算子的具体实现。

## 0. 术语纠正（重要）

S60 是**燧原 Enflame GCU**，其运行时事件名是 `tops*`（`topsLaunchKernel` /
`topsModuleLaunchKernel` / `topsStreamSynchronize` 等），厂商算子库**不是 CNNL**——
**CNNL 是寒武纪 MLU 的算子库**。仓库历史里若干 S60 相关文档（含本 campaign 的
round artifact 与 flexattention/fused_moe 的 s60 final_summary）沿用了「CNNL」，
这是**历史性笔误**，实指「GCU 厂商库（TOPS runtime）」；阅读时按此理解，本文及
更新后的 README/summary 已改用「GCU 厂商库」表述。

## 1. 硬性 Triton 约束（probe-backed，写进 `triton_gcu` profile）

这些是 triton_gcu 3.6.0 在 S60 上的**编译期硬约束**，违反即 FAIL，任何算子都要遵守：

| 约束 | 具体表现 | 后果 |
|---|---|---|
| `tl.dot` 要求 **2 的幂** | M/N/K 仅 `16/32/64/128` 通过；`48/80/96/112/160/192` 全 FAIL（注意 `96=16×6` 也失败，证明是 2 的幂而非 16 倍数） | T=83 只能 pad 到 128，58% FLOP 浪费 |
| `tl.arange` 要求 **2 的幂** | 同 `tl.dot`；`83/96/100/120` 全 FAIL | 循环/tile 长度必须取 2 的幂 |
| `tl.max` / `tl.sum` **不支持 `keepdim`** | 标准 triton 支持 `keepdim`，GCU 版不支持 | 用 `axis=1` + `[:, None]` 广播替代 |
| `tl.dot` 要求**两操作数同 dtype** | fp32 × fp16 报错 | 统一 cast（如 `v.to(tl.float32)`） |
| `num_warps` 合法值 `1/2/4/8` | 均可编译执行 | fp16 dot 下 `num_warps=1` 最优，`8` 严重退化（~2x） |
| `tl.atomic_add` **不可用** | 最小 atomic 例子也 `Pipeline run failed: PassManager execution failed` | **跨 program 归约（segment-max/sum、scatter）无法融合进 kernel，必须 host `torch.sum`/`torch.max`**；这是 sparse_pooler segment-max 与 mhc_head_compute_mix_backward 两归约无法融合的结构性原因 |
| `tl.log1p` / `tl.maximum` 支持情况 | `tl.log1p` 无，`tl.maximum` 有 | log1p(relu) 用 `tl.log(1.0 + tl.maximum(x, 0.0))` 实现 |
| 设备侧 profiler **只有 launch 事件** | 无 `cat=kernel` device-duration | device 时间靠 `wall − launch-API` 反推 |

## 2. S60 算子分两类：launch-bound 融合赢，device-bound 手写输

这是 epoch2 六个算子的核心结论：

### 2a. launch-bound（base 海量小 launch → 融合能赢 base）

base 是海量小 launch 时（每 launch 数据量小、host 调度开销占主导），手写单 kernel
融合省 launch 的收益远超 device 惩罚。epoch2 兑现：

- `centre_random_augmentation`：**96→10 launch，1.90x**（四元数→R + 3×3 matvec + 平移
  + mask 全融合单 kernel，host 只留随机数）。这是 S60 首个打赢 base 的算子。
- `music_flamingo_rotary_embedding`：**13→3 launch，1.11x**（freqs elementwise 融合单
  kernel，cos/sin 保留 vendor）。
- （epoch1 已兑现的 fused_moe 13.8x、groupedtopk 1.68x 同属此类）

### 2b. device-bound（厂商库 GEMM/attention → 手写输给库）

base 落到 GCU 厂商库（TOPS runtime 张量核心）时，手写即便用上 `tl.dot` 也受 2 的幂
约束（T=83→pad 128，58% FLOP 浪费）+ launcher 税仅 17.4us（图回放无收益）。epoch2 交付
「比 epoch1 强」但 <1x：

- `mm_encoder_attention`：fp16 dot，0.27x→0.92x（3.4x over epoch1）
- `flexattention`：causal fp16 dot，0.42x→0.94x（2.2x over epoch1）
- `mhc_post_layer_mix`：BLOCK_H+bf16，0.56x→0.77x（+37% over epoch1）
- `sparse_pooler`：确认 measurement-bound（GEMM 61% 库 + 手写 segment-max 慢 4x）

## 3. 交付标准：比 epoch1 强，而非打赢 base

- **正确标准**：交付「相对 epoch1 更强的 Triton 算子」即可，不必打赢厂商库。
- 能赢 base 的是 launch-bound 融合（centre_random_augmentation 1.90x、music_flamingo 1.11x）；
  device-bound 的 attention/GEMM 手写只能拿 3~4x over epoch1 但仍 <1x。
- **不要**用「打不赢 base」判 terminal——比 epoch1 强就是合格交付。

## 4. 可复用的优化公式

### 4a. launch-bound 全融合（赢 base 的关键）

1. **先 profile 数 launch**：base 的 `topsLaunchKernel`/call >10 且数据量小 → launch-bound。
2. **全融合**：把 elementwise + 小矩阵运算全塞进单 kernel，host 只留必须的（随机数、库算子）。
3. **部分融合**（关键教训）：**vendor 库算子（cos/sin 三角、GEMM）保留**，只融合周围的
   elementwise——epoch1 全融合用 tl.cos/tl.sin 反而 -13%（GCU math-dialect 三角慢 44%），
   music_flamingo 保留 vendor cos/sin 后 1.11x。

### 4b. device-bound attention/GEMM（比 epoch1 强的修复）

1. **切 fp16 `tl.dot`**：QK^T 保留 fp16 输入（不 widen），比 fp32 快 ~37us。
2. **收并行粒度**：1 program/token → 1 program/(batch,head)。
3. **去布局拷贝**：`.contiguous()` → 0（strided addressing）。
4. **tl.dot 只用于大 tile**：K=4 的收缩（mhc_post_layer_mix）用 tl.dot 是灾难（0.019x），
   小收缩用 tl.sum 展开更好。

## 5. 六个算子的最终结论

| 算子 | epoch1 | epoch2 | 类型 |
|---|---:|---:|---|
| centre_random_augmentation | 0.95x | **1.90x** ✅ | launch-bound 全融合 |
| music_flamingo_rotary_embedding | 0.9x | **1.11x** ✅ | launch-bound 部分融合 |
| mm_encoder_attention | 0.27x | 0.92x | device-bound attention |
| flexattention | 0.42x | 0.94x | device-bound attention |
| mhc_post_layer_mix | 0.56x | 0.77x | device-bound 小 GEMM |
| sparse_pooler | 0.79x | 0.79x（measurement-bound） | GEMM 61% 库占优 |

## 6. profile 资产

- `skills/kernel-opt-loop/profiles/triton_gcu/` 已建立完整 machine-readable profile（v1），含 2 条 approved probe evidence（`tl.dot` power-of-2、`num_warps 1/2/4/8`）。
- 后续 S60 campaign 直接复用该 profile，无需重新 onboarding。

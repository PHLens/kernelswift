# BI150（Iluvatar BI-V150）Triton 优化总结

本文总结 track1 中 **全部 10 个算子在 BI150 后端上的 Triton 优化结果**、BI150 平台的特殊性，以及可复用的优化经验。

> 说明：track1 共 10 个算子，BI150 后端已全部覆盖。task 1（groupedtopk）于更早的 PR 完成（`torch.compile` 路线）；task 5–10 为第一批 Triton kernel-fusion 优化；task 2–4（flexattention / fused_moe / sparse_pooler）为第二批补做。

## 一、总体成果

| 算子 | 优化结果 | 手段 | 关键数字 |
|---|---|---|---|
| groupedtopk | +41.6%（1.71x） | `torch.compile(mode="reduce-overhead")`（非 Triton，见下） | wall 0.475→0.277 ms |
| flexattention | 无优化空间 | naive Triton causal attention 交付物（正确，0.612x） | base 为厂商 Ixmma FlashAttention（CausalM_t=2） |
| fused_moe | +84.9%（6.60x） | per-expert dispatch 融合 + tl.dot GEMM 融合 | 123.9→9.82 kernels/call |
| sparse_pooler | +17.0%（1.22x） | log1p(relu) + per-sequence max-pooling 融合 | 11.92→6.88 kernels/call |
| music_flamingo_rotary_embedding | +48.64%（1.95x） | kernel fusion（13 elementwise → 1） | 13→1 kernels/call |
| mm_encoder_attention | 无优化空间 | naive Triton attention 交付物（正确，0.547x） | base 为厂商 Ixmma FlashAttention（CausalM_t=0） |
| mhc_post_layer_mix | +20.09%（1.20x） | elementwise 尾融合（GEMM 不动） | 5.66→2.96 kernels/call |
| mhc_head_compute_mix | +87.17%（7.79x） | Sinkhorn 20 轮迭代 + elementwise 全融合 | 132.88→1.0 kernels/call |
| centre_random_augmentation | +77.7%（4.49x） | 确定性计算全链融合（RNG 留 host） | 78.8→5.52 kernels/call |
| mhc_head_compute_mix_backward | +43.11%（1.76x） | sigmoid backward + 两个 reduce 融合 | 9.74→2.96 kernels/call |

（groupedtopk 的 +41.6% 走 `torch.compile(mode="reduce-overhead")` 路线，非 Triton kernel 融合——其 Triton 直接重写因 `torch.topk` tie 语义锁死而放弃。）

## 二、BI150 平台的特殊之处

### 1. 库 kernel 深度优化，Triton 难以超越「厂商手写 kernel」

BI150 的 PyTorch 栈（CoreX 发行版）把许多算子 dispatch 到**厂商专门调优的 CUDA kernel**：
- `torch.topk` → `gatherTopK` / `bitonicSortKVInPlace`（groupedtopk 的瓶颈，占 device 67%）
- `F.scaled_dot_product_attention` → `FlashAttnFwdF16Ixmma`（attention 的整条链，单 kernel）
- `torch.einsum` → `gemm_tcu_h`（TCU 批 GEMM，经 cublasLt 接口）

这些 kernel 用 Ixmma tensor-core 指令和 TCU 硬件单元，是厂商为 BI150 专门调优的。Triton 重写要超越它们**通常不现实**：
- groupedtopk 的 `torch.topk` tie 语义锁死了直接重写（早期 Round 002 因平局序错失败，最终走 torch.compile）
- attention 的 Ixmma FlashAttention 无法超越（最终只出正确交付物，不追求超越）
- post_layer_mix 的窄 GEMM `[4,4]@[4,1280]` 因 contraction 维太小，tl.dot tile 严重不匹配

**经验**：先通过 profiler 确认算子是否已经走厂商库 kernel；若是，Triton 优化的预期要保守，优先融合「库 kernel 之外的 elementwise 尾巴」，而不是硬碰库 kernel。

**关键补充（fused_moe vs sparse_pooler 的对比）**：同样是含 GEMM 的算子，结局因「瓶颈类型」而异：
- **fused_moe（launch-bound）**：GEMM 的 M 维小（per-expert，M≈20）且 fp16，但 123.9 个 kernel 的 launch 冗余是主导。用 `tl.dot` 融合 GEMM + 消除 dispatch 冗余 → **+79.98%**（tl.dot 虽比 TCU 慢，但省下的 launch 开销远超 GEMM 的损失）。
- **sparse_pooler（compute-bound）**：GEMM 的 N 维大（decoder 768×30522）且 fp32，只有 2 个 GEMM launch，无 launch 冗余。要赢只能「算得比 TCU 快」，而 fp32 大 GEMM 的 tl.dot 大概率不能映射到厂商 TCU → **vendor-optimal-bound，stop**。

**结论**：tl.dot 融合 GEMM 的价值取决于「是否有 launch 冗余可省」，而非「tl.dot 是否比 TCU 快」。launch-bound 场景 tl.dot 融合是大赢家，compute-bound 场景则无空间。

### 2. 小 shape 算子的 host-bound 是普遍现象

BI150 上小 shape 算子的 wall time 常被 host/launch 开销主导：
- rotary baseline：device_ratio 0.194（~80% host）
- attention：device_ratio 0.099（~90% host，且是 harness 固定的 set_seed + synchronize）
- centre_random baseline：device_ratio 0.392
- 多数算子融合后 device_ratio 降到 0.07–0.12（强 host-bound）

**经验**：
- host-bound 不是「没有优化空间」——kernel fusion 减少 launch 次数能**同时**压缩 device 时间和 host launch 开销，wall 收益常大于纯 device 收益。
- 但当 device_ratio 降到 0.1 以下、且剩余 host 是 harness 固定的 `set_seed`/`cuda.synchronize` 时，就到了 measurement-bound 终点（skill 的 stop 判定）。

### 3. Triton 原语在 BI150 上的能力矩阵（需实测，勿假设 NVIDIA）

`triton_cuda` profile 最初把多个原语标为 Unknown，本项目通过实测逐步确证：

| 原语 | 实测结果 | 证据 |
|---|---|---|
| `tl.dot` | **Supported**（fp32/bf16 (32,32) 精确；fp16 收缩 128/64 正确 lower，max_rel_err ~2e-4；**M≥16 warp tile 约束**，M=1/2/4 无法 lower） | `scripts/bi150_tl_dot_probe*.py`；fused_moe Round 002 |
| `tl.sqrt/sin/cos` | **Supported**（与 torch 逐位一致，max_abs_diff=0.0） | centre_random Round 002 |
| `tl.sigmoid` / `tl.sum` / `tl.atomic_add` | **Supported** | backward Round 001 |
| `tl.static_range` | 小迭代（4 次）可用，**大迭代（19 次）编译爆炸**（>300s） | head_compute_mix |
| `num_warps` / `num_stages` / block pointers | 仍 Unknown | — |

**经验**：
- BI150 是 `cuda` backend 的 CoreX 发行版，NVIDIA 的经验**部分适用但不保证**，关键原语（尤其 `tl.dot`、超越函数）必须**文件化 probe 实测**，不能假设。
- `tl.static_range` 大循环展开会导致编译时间爆炸，动态 `tl.range` 是安全替代（语义等价）。

### 4. 随机算子的正确性约束：RNG 必须保留 host 侧

centre_random_augmentation 含随机数（四元数 + 平移）。harness 每次 forward 前 `set_seed`，base 和 candidate 各自重播同一 seed，因此：

- **RNG 消耗顺序/数量/分布必须与 base 逐位一致**（3×torch.rand + 1×torch.randn，同顺序）
- 正确做法：**随机数生成保留在 host 侧 torch 调用，只融合确定性计算**（这是 zero-risk 的策略）
- 若试图在 Triton kernel 内复刻 torch.rand 的分布（Philox），风险极高且收益不明

## 三、可复用的优化经验（按价值排序）

### 1. Kernel fusion 是 BI150 上收益最大的手段

| 算子 | 融合前 kernels/call | 融合后 | wall 提升 |
|---|---|---|---|
| head_compute_mix（Sinkhorn） | 132.88 | 1.0 | +87.17% |
| fused_moe（per-expert 循环 + GEMM） | 123.9 | 9.82 | +84.9%（累计） |
| centre_random（刚体变换） | 78.8 | 5.52 | +77.7% |
| rotary（elementwise） | 13 | 1 | +48.64% |
| backward（sigmoid+reduce） | 9.74 | 2.96 | +43.11% |
| post_layer_mix（elementwise 尾） | 5.66 | 2.96 | +20.09% |
| sparse_pooler（激活 + pooling） | 11.92 | 6.88 | +17.0% |

**规律**：kernel 数量越多、每个 kernel 越小，融合收益越大。Sinkhorn 迭代（132 kernel）和刚体变换（78 kernel）是极端案例。

### 2. 迭代型算子（Sinkhorn 等）是融合的黄金目标

head_compute_mix 的 Sinkhorn 20 轮迭代产生了 ~120 个极小 kernel（每个只处理 256 元素），融合成单 kernel 后 +87.17%。**任何带 Python 循环的算子都要检查循环体是否在 device 上产生了重复的 kernel launch**。

### 3. 随机数留在 host，确定性计算融合进 kernel

含随机数的算子（centre_random），把 RNG 留 host、只融合确定性部分，是零正确性风险的通用模式。

### 4. 分步融合降低风险

centre_random 分两步：Round 001 先融合「大块确定性计算」，Round 002 再融合「超越函数链」。每轮验证正确性，逐步降低风险。超越函数（sqrt/sin/cos）的融合要单独 probe 验证数值一致性（虽然 BI150 实测与 torch 逐位一致，但不假设）。

### 5. 库 kernel 不硬碰，融合「尾巴」

post_layer_mix 的 GEMM 走 TCU 批 GEMM（厂商库），Round 001 只融合 GEMM 之后的 elementwise 尾（+20.09%），Round 002 判断窄 GEMM 无法被 tl.dot 超越而 stop。**「库 kernel 保持不动，融合周边的 elementwise」是稳妥且收益明确的选择**。

**fused_moe 的两步走 pattern**：Round 001 用 `torch.argsort` 把 token 按 expert 分桶（让 torch GEMM 能用连续 slice），替代 8 次 CUB 选择；但 argsort 本身成为新瓶颈（107 us）。Round 002 把 GEMM 融合进 Triton（tl.dot）后，argsort 的依赖自然消除。**「先用分桶/排序优化 dispatch，再把 GEMM 融合进 kernel 消除分桶依赖」是处理 per-expert/per-group 循环算子的通用两步走**。

### 6. measurement-bound 的诚实判定

融合到单 kernel 后，device_ratio 降到 0.07–0.12，剩余 wall 是 harness 固定的 `set_seed` + `cuda.synchronize`。此时诚实判定 measurement-bound 并 stop，而不是硬编没有意义的优化。这符合 skill 的 bottleneck-judgment 规则，也避免浪费时间。

## 四、关键基础设施产出（对后续后端可复用）

1. **`triton_cuda` profile**（`skills/kernel-opt-loop/prompts/coder_targets/triton_cuda.md`）：从「大量 Unknown」逐步确证到 tl.dot / tl.sqrt/sin/cos / tl.sigmoid / tl.sum / tl.atomic_add 都 Supported。
2. **probe 脚本**（`scripts/bi150_*`）：tl.dot、torch.compile、triton smoke 等 probe，可复用于其它算子/后端。
3. **`make_baseline_adapter.py` 的 super() 修复**：重命名 `Model`→`ModelNew` 时同步重命名 `super(Model, ...)` 里的类名引用，避免 `NameError`。
4. **随机算子处理模式**：RNG 留 host 的决策模板（见 centre_random 的 decision）。

## 五、总结一句话

BI150 上 Triton 优化的核心是 **kernel fusion（把 device 上重复的小 kernel 合并成单个 Triton kernel）**，收益与 kernel 数量正相关；同时要**尊重厂商库 kernel（不硬碰）、实测原语能力（不假设 NVIDIA）、诚实判定 measurement-bound（不硬编）**。

# S60（燧原 GCU）Triton 优化经验沉淀

> 来源：`mm_encoder_attention` S60 epoch2 campaign（0.27x → 0.92x，两个 round 收敛）。
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
| 设备侧 profiler **只有 launch 事件** | 无 `cat=kernel` device-duration | device 时间靠 `wall − launch-API` 反推 |

## 2. S60 是 device-bound 后端

- base 的 `F.scaled_dot_product_attention` 落到 GCU 厂商库（TOPS runtime）的 flash-attention kernel，device floor ~158us；
- S60 的 **launcher 税仅 ~17.4us**（对比 BI150 的 ~84.77us），host 链（view/transpose/reshape）仅 ~11us；
- 结论：**图回放（graph-replay）在 S60 上无收益**——它在 BI150 能翻盘是因为 BI150 的 launcher 税是 S60 的 5 倍。

## 3. 交付标准：比 epoch1 强，而非打赢 base

这是本次最重要的方法论纠正：

- **正确标准**：交付一个「相对 epoch1 更强的 Triton 算子」即可，不必打赢厂商库。
- `mm_encoder_attention`：epoch1 0.27x（`tl.sum` 标量展开 + 1328 program + 3×`.contiguous()`）→ epoch2 0.92x（fp16 `tl.dot` 单 tile + 16 program + 零 `.contiguous()`），**3.4x 提升**，是明确成功。
- 即使「不赢 base」（device-bound 撞 GCU 厂商库），只要比 epoch1 强，就是合格交付，**不要**用「打不赢 base」判 terminal。

## 4. 可复用的优化公式（attention/GEMM 类）

从 epoch1 到 epoch2 的三步修复，对 S60 上其它 attention/GEMM 算子同样适用：

1. **切 `tl.dot`**：epoch1 误判「tl.dot Unknown」而用 `tl.sum` 标量展开。实际 tl.dot 可用（2 的幂约束），两个 GEMM 都切到 tl.dot；QK^T 用 **fp16 tensor core**（保留 fp16 输入不 widen，比 fp32 快 ~37us）。
2. **收并行粒度**：1 program/token（1328 个，重复 load 全 K/V）→ 1 program/(batch,head)（16 个，整 tile 进寄存器）。
3. **去布局拷贝**：`3× .contiguous()` → 0（直接 strided addressing `[B,S,H*D]`）。

## 5. 后续算子的直接提示

- `flexattention`（0.42x）：同样 attention 病根，用上述公式预计 ~3x 级提升（即使不赢 base）。causal 场景注意 mask 上三角，其余约束同 mm_encoder。
- `mhc_post_layer_mix`（0.56x）：**不用 tl.dot**（K=4 收缩维太小，2 的幂约束下 dot 无意义），C500 的 31.66x 模板（4 次显式 fp32 MAC + elementwise 折叠）仍适用，不受本约束影响。
- `groupedtopk`（1.68x 已 accepted）：reduction 类，无 GEMM，不受 dot 约束影响。

## 6. profile 资产

- `skills/kernel-opt-loop/profiles/triton_gcu/` 已建立完整 machine-readable profile（v1），含 2 条 approved probe evidence（`tl.dot` power-of-2、`num_warps 1/2/4/8`）。
- 后续 S60 campaign 直接复用该 profile，无需重新 onboarding。

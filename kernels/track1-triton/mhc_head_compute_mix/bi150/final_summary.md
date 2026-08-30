# mhc_head_compute_mix（BI150）最终总结

## 结论

✅ **完结 · 7.79x**（r001，`1.433128 → 0.183889 ms`）；r002 measurement-bound，stopped

## 轮次摘要

| Round | 干预 | 结果 | Wall (ms) | Device (us/call) | Kernel count |
|---|---|---|---|---|---|
| 000 | Phase 0 baseline | baseline | 1.517299 | 926.395 | 132.88 |
| 001 | kernel-fusion（Sinkhorn 20 轮迭代 + elementwise 融合为单 kernel） | accepted | 0.183889 | 12.996 | 1.0 |
| 002 | 无干预（host-bound，device 已最优） | aborted | - | - | - |

## 关键证据

- **瓶颈分类**：baseline device-bound（device_ratio 0.61），~120/133 kernel 是 Sinkhorn 迭代产生的小 kernel（20 轮 × 每轮 sum/div/add-eps，作用在 [16,4,4] 256 元素上）
- **Round 001 收益**：**+87.17%** wall（`1.433128 → 0.183889 ms`），kernel_count `132.88 → 1.0`，device `924.79 → 12.996 us/call`。这是所有算子中融合收益最大的一次
- **融合内容**：2 个 sigmoid + 1 个 stable softmax exp + 20 轮行列交替 Sinkhorn 归一化，全部融合进单个 `_mhc_head_compute_mix_kernel`（grid=16，[4,4] tile 全程寄存器驻留，tl.sum 寄存器内归约）
- **正确性**：harness `PASS accuracy`，独立 probe max_abs_diff：pre=5.96e-08, post=0.0, comb=1.19e-07（Sinkhorn eps 非对称放置精确验证）

## 关键实现细节

- 决策指定的 `tl.static_range(19)` 编译期展开在 BI150 上不可行（19 次展开编译 >300s），Coder 降级为 `tl.range(19)` 动态循环，语义等价
- eps 非对称放置精确保留：首轮显式行归一 `comb/sum + eps`（eps 加到矩阵）vs 循环内/列归一 `comb/(sum + eps)`（eps 加到分母）

## 停止边界

r001 后 device_ratio 0.0755（~92.5% wall 是 harness-fixed host 开销：set_seed + cuda.synchronize）。device 已融合到单 kernel（kernel_count 1.0），作用于极小 [16,4,4] 张量，无冗余 device 数据流可再压。r002 Designer 判定 measurement-bound，stopped。

## 产物

- canonical: `triton_mhc_head_compute_mix_001.py`
- 决策/报告: `rounds/decision_001.md`、`rounds/coder_result_001.md`、`rounds/report_001.md`、`rounds/decision_002.md`
- 状态: `team-state.md`（`phase: stopped`，`stop_reason: user-intervention`）

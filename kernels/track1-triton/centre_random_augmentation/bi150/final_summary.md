# centre_random_augmentation（BI150）最终总结

## 结论

✅ **完结 · 4.49x**（累计，`1.073250 → 0.239284 ms`）；r003 measurement-bound，stopped

## 轮次摘要

| Round | 干预 | 结果 | Wall (ms) | Device (us/call) | Kernel count |
|---|---|---|---|---|---|
| 000 | Phase 0 baseline | baseline | 1.073250 | 420.684 | 78.8 |
| 001 | kernel-fusion（centering + rot_vec_mul + 平移 + mask） | accepted | 0.712600 | 237.95 | 54.8 |
| 002 | kernel-fusion（四元数→旋转矩阵的超越函数 sqrt/sin/cos） | accepted | 0.239284 | 29.24 | 5.52 |
| 003 | 无干预（measurement-bound） | aborted | - | - | - |

## 关键证据

- **瓶颈分类**：mixed（baseline device_ratio 0.392，偏 host-bound），输出仅 [4,256,3]（3072 元素）但 forward 启动 ~79 个 tiny kernel
- **Round 001 收益**：+30.35% wall，kernel_count 78.8 → 54.8（融合 centering + rot_vec_mul + 平移 + mask）
- **Round 002 收益**：+66.37% wall，kernel_count 54.8 → 5.52（融合四元数→旋转矩阵的 sqrt/sin/cos/mul/add/stack/cat）
- **累计**：4.49x（77.7% 提升），kernel count 78.8 → 5.52

## 关键技术发现（对后续算子有价值）

1. **tl.sqrt/sin/cos 在 BI150 上可 lower 且与 torch 逐位一致**（max_abs_diff=0.0，经文件化探针验证）。此前 profile 标记 Unknown，现已证实。这解除了「超越函数不能融合进 Triton kernel」的疑虑。
2. **随机数边界策略**：含随机数的算子，把 RNG（torch.rand/randn）保留在 host 侧、按原顺序，只融合确定性计算。harness 每次 forward 前 set_seed，RNG 顺序一致 → R/T 逐位一致 → 正确性零风险。这是处理随机算子的通用模式。
3. **分步融合**：先融合「大块确定性计算」（Round 001），再融合「超越函数链」（Round 002），逐步降低风险、逐轮验证。

## 停止边界

r002 后 device_ratio 0.122，确定性计算已完全融合进单个 `_centre_aug_kernel`（6.81 us/call）。剩余 5.5 kernel 是：host RNG（3×rand + 1×randn，硬不变式）+ 1 个 s_trans=1.0 的 no-op mul + stray cat。唯一可改的 no-op mul 只省 <1.5% wall，低于 5% 阈值。r003 Designer 判定 measurement-bound，stopped。

## 产物

- canonical: `triton_centre_random_augmentation_002.py`
- 决策/报告: `rounds/decision_001/002/003.md`、`rounds/coder_result_001/002.md`、`rounds/report_000/001/002.md`
- 状态: `team-state.md`（`phase: stopped`，`stop_reason: user-intervention`）

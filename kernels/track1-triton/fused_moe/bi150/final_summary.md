# fused_moe（BI150）最终总结

## 结论

✅ **完结 · 6.60x**（累计，`3.258671 → 0.493474 ms`）；r003 measurement-bound，stopped

## 轮次摘要

| Round | 干预 | 结果 | Wall (ms) | Device (us/call) | Kernel count |
|---|---|---|---|---|---|
| 000 | Phase 0 baseline | baseline | 3.258671 | 968.162 | 123.9 |
| 001 | kernel-fusion（per-expert dispatch/gather/scatter 融合，argsort 分桶） | accepted | 2.488731 | 504.312 | 54.1 |
| 002 | gemm-fusion（tl.dot 融合 per-expert GEMM + 消除 argsort） | accepted | 0.493474 | 140.84 | 9.82 |
| 003 | 无干预（measurement-bound） | aborted | - | - | - |

## 关键证据

- **瓶颈分类**：baseline mixed（device_ratio 0.297），123.9 kernel/call 由 per-expert Python 循环（8 次）产生
- **Round 001 收益**：+21.44% wall，kernel_count 123.9 → 54.1（用 argsort 分桶 + 单 kernel 加权归约替代 CUB DeviceSelect/gather/scatter）
- **Round 002 收益**：+79.98% wall，kernel_count 54.1 → 9.82（tl.dot 融合 per-expert GEMM + chunk + SiLU + 加权归约，消除 argsort）
- **累计**：6.60x（84.9% 提升）

## 关键技术发现（对后续算子有价值）

1. **tl.dot 大收缩维在 BI150 可用**：fp16 收缩 128/64 正确 lower（max_rel_err ~2e-4），但 **M≥16 warp tile 约束**（M=1/2/4 无法 lower），因此采用 per-expert 批量布局（BLOCK_M=256）。这扩展了 profile 的 tl.dot 记录（原来只有 (32,32)）。
2. **argsort 分桶是过渡手段**：Round 001 用 argsort 让 torch GEMM 能用连续 slice，但 argsort 本身成为新瓶颈（107 us）。Round 002 把 GEMM 融合进 Triton（tl.dot）后，argsort 依赖消除，这是「先分桶优化 dispatch，再融合 GEMM 消除分桶」的两步走。
3. **topk tie 语义硬约束**：`torch.topk` 始终保留（不重写），tie 语义逐位继承，规避 groupedtopk 的失败模式。

## 停止边界

r002 后 device_ratio 0.2854（~71% host/launch），剩余 device（140.84 us）中 topk（39.44 us，tie 硬约束不可碰）+ fused kernel（55.8 us，已单 kernel 最优）占 68%，小开销（cast/renormalize/zero-init）每项 <5% wall 且在 launch-bound 下 device 节省不 1:1 映射。r003 Designer 判定 measurement-bound，stopped。

## 产物

- canonical: `triton_fused_moe_002.py`
- 决策/报告: `rounds/decision_001/002/003.md`、`rounds/coder_result_001/002.md`、`rounds/report_000/001/002.md`
- 状态: `team-state.md`（`phase: stopped`，`stop_reason: user-intervention`）

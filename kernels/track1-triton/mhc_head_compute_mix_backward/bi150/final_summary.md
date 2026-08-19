# mhc_head_compute_mix_backward（BI150）最终总结

## 结论

✅ **完结 · 1.76x**（r001，`0.351449 → 0.198597 ms`）；r002 measurement-bound，stopped

## 轮次摘要

| Round | 干预 | 结果 | Wall (ms) | Device (us/call) | Kernel count |
|---|---|---|---|---|---|
| 000 | Phase 0 baseline | baseline | 0.351449 | 185.599 | 9.74 |
| 001 | kernel-fusion（sigmoid backward 全链 + 两个 reduce 融合为单 kernel） | accepted | 0.198597 | 14.692 | ~2.96 |
| 002 | 无干预（measurement-bound） | aborted | - | - | - |

## 关键证据

- **瓶颈分类**：baseline device_ratio 0.528，device 被两个 sum reduce 主导（`reduce_kernel<sum_functor>` 147.98 us/call ≈ 80% device）
- **Round 001 收益**：+43.11% wall（`0.349112 → 0.198597 ms`），device `186.057 → 14.692 us/call`（−92%），kernel_count `9.74 → 2.96`
- **融合内容**：单 kernel 完成 z=im*scale+base → sigmoid → grad_z=go*σ*(1-σ) → grad_input_mix，并在寄存器内完成两路归约（grad_mhc_base 用 tl.sum(axis=0) + atomic_add 得 [4]，grad_mhc_scale 用全量 tl.sum + atomic_add 得 [1]）
- **正确性**：harness `PASS accuracy`，独立 probe max_abs_diff：gim=1.49e-8, gs=1.67e-6, gb=1.14e-5（两个 reduce 维度语义精确验证）

## 停止边界

r001 后 device_ratio 0.074（强 host-bound，~93% wall 是 harness-fixed 的 set_seed + cuda.synchronize）。device 已到单 kernel 理论最小（7.416 us/call，输入仅 8192 元素）。torch.zeros 累加器初始化（7.276 us/call）占 wall 仅 3.7%，且消除它违反 per-call fresh-tensor 语义 + input-not-mutated 不变式。r002 Designer 判定 measurement-bound，stopped。

## 产物

- canonical: `triton_mhc_head_compute_mix_backward_001.py`
- 决策/报告: `rounds/decision_001/002.md`、`rounds/coder_result_001.md`、`rounds/report_000/001.md`
- 状态: `team-state.md`（`phase: stopped`，`stop_reason: user-intervention`）

# mm_encoder_attention（BI150）最终总结

## 结论

📦 **Triton 交付物就绪（正确，未超越 base）** · 优化层面 measurement-bound（host-bound 锁死）

- 产出正确的 naive Triton attention（`triton_mm_encoder_attention_001.py`），harness PASS accuracy，0.547x（比 base 慢，符合预期）
- 优化层面无 ≥5% 可证伪干预：base 是厂商 Ixmma FlashAttention，且 ~90% wall 是 harness-fixed host 开销

## 轮次摘要

| Round | 干预 | 结果 | Wall (ms) | Device (us/call) | Kernel count |
|---|---|---|---|---|---|
| 000 | Phase 0 baseline | baseline | 0.151139 | 14.949 | 0.86 |
| 001 | 无干预（tl.dot 当时标记 Unknown） | aborted | - | - | - |
| 002 | 无干预（tl.dot 已 Supported，但 host-bound 锁死） | aborted | - | - | - |
| - | **参赛交付物**：naive Triton attention | **正确（0.547x）** | 0.358623 | - | - |

## 为什么优化层面无空间（最终结论）

### 1. base 是厂商最优的融合 FlashAttention kernel

`base.py` 仅调用 `F.scaled_dot_product_attention`，实测 dispatch 到单 `FlashAttnFwdF16Ixmma<128,128,16,64,64,Causal=0,Alibi=0>` kernel（Iluvatar Ixmma tensor-core）。QK^T + softmax + PV 已融合在单 kernel，无碎片可优化。

### 2. host-bound + harness-fixed（决定性约束）

wall = 151 us，device 仅 14.949 us（device_ratio ≈ 0.099）。~90% wall 是 harness 的 `set_seed` + `cuda.synchronize()` + 输入 clone，在 `ModelNew.forward` 和 candidate 边界之外。**即使 Triton attention 设备侧完美零成本，也只能省 14.949/151 ≈ 9.9%，且需不增加任何 JIT/launch 开销才能过 5% 阈值**——不可实现。

### 3. tl.dot 已 Supported，但不改变瓶颈

Round 001 abort 时的「tl.dot Unknown」前提已被推翻（Orchestrator 实测 fp32 精确、bf16 near-exact，profile 已更新为 Supported）。但移除 capability-miss 后，瓶颈仍是 host-bound：naive 分解版会拆散融合 kernel 严格更慢，flash 版要打赢厂商调优的 Ixmma kernel 且 tl.dot 无性能证据（只验证了正确性）。

## 参赛交付物（正确性优先，非优化）

按比赛要求「提交 Triton 代码」，产出了正确的 naive Triton attention：

- **candidate**: `triton_mm_encoder_attention_001.py`（SHA `88ade697...`）
- **结构**: 单 `@triton.jit` kernel，grid=(B*H,)，每 program 处理一个 (batch,head)；QK^T（`tl.dot`）+ softmax（`exp(x-max)` 稳定形式）+ PV（`tl.dot`）；fp16 输入转 fp32 全程计算，输出 cast 回 fp16；BLOCK_S=128 padding + mask
- **验收**: harness `PASS accuracy`（atol=1e-2, rtol=1e-2, equal_nan=True）
- **速度**: 0.547x（慢于 base 符合预期，base 是厂商 Ixmma FlashAttention）

## 产物

- canonical（优化）: `baseline_adapter.py`（优化层面未产生 accepted candidate）
- 参赛交付物: `triton_mm_encoder_attention_001.py`
- 决策/报告: `rounds/decision_001.md`（abort）、`rounds/decision_002.md`（abort）、`rounds/report_000.md`、`rounds/coder_result_deliverable.md`
- 状态: `team-state.md`（`phase: stopped`，`stop_reason: user-intervention`）

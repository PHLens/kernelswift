# sparse_pooler（BI150）最终总结

## 结论

✅ **完结 · 1.22x**（r001，`1.060573 → 0.880377 ms`）；r002 vendor-optimal-bound，stopped

## 轮次摘要

| Round | 干预 | 结果 | Wall (ms) | Device (us/call) | Kernel count |
|---|---|---|---|---|---|
| 000 | Phase 0 baseline | baseline | 1.070492 | 743.064 | 11.92 |
| 001 | activation-pooling-fusion（log1p(relu) + per-sequence max-pooling 融合） | accepted | 0.880377 | 609.397 | 6.88 |
| 002 | 无干预（vendor-optimal-bound） | aborted | - | - | - |

## 关键证据

- **瓶颈分类**：baseline compute-bound（device_ratio 0.694），GEMM 主导（dense + decoder 走 TCU，~78% device）
- **Round 001 收益**：+16.99% wall（`1.060573 → 0.880377 ms`），kernel_count `11.92 → 6.88`，device `743.80 → 609.40 us/call`
- **融合内容**：decoder 后的 log1p(relu) + 4×max-pooling 融合为单 `_sparse_pooler_fused_kernel`，不物化 [83,30522] 中间 tensor；额外收益是移除了 `seq_lens.tolist()` 的 D2H 同步（50× Memcpy + 50× synchronize）
- **正确性**：harness `PASS accuracy`（list 输出逐元素比较），独立 probe max_abs ≈ 1.19e-07（近 bit 精确）

## 停止边界（vendor-optimal-bound）

r001 后剩余 92.5% device 是两个 TCU GEMM（`gemm_tcu_h` 482.53 + `GEMM_Epilogue` 81.29 ≈ 563.8 us/call），已在厂商 tensor-core 硬件上。这是 **compute-bound 而非 launch-bound**（仅 2 个 GEMM launch，无 launch 冗余可省），tl.dot 重写需「在 fp32 大 N（收缩 768，N=30522）上超越厂商调优的 TCU」，而 profile 只证明 (32,32)@(32,32)，fp32 大 GEMM 的 tl.dot 未证明且大概率不能映射到厂商专有 TCU。r002 Designer 判定 vendor-optimal-bound（非 measurement-bound，device 工作真实存在但不可压缩），stopped。

## 与 fused_moe 的对比（同含 GEMM + 循环，但结局不同）

| | fused_moe | sparse_pooler |
|---|---|---|
| 瓶颈类型 | launch-bound（123.9 kernel） | compute-bound（11.92 kernel） |
| GEMM 规模 | 小 M（per-expert，M≈20）fp16 | 大 N（decoder 768×30522）fp32 |
| tl.dot 融合 | 成功（+79.98%，消除 launch 冗余） | 不适用（无 launch 冗余，需算得比 TCU 快） |
| 最终 | 6.60x | 1.22x |

## 产物

- canonical: `triton_sparse_pooler_001.py`
- 决策/报告: `rounds/decision_001/002.md`、`rounds/coder_result_001.md`、`rounds/report_000/001.md`
- 状态: `team-state.md`（`phase: stopped`，`stop_reason: user-intervention`）

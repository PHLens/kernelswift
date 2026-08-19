# flexattention（BI150）最终总结

## 结论

📦 **Triton 交付物就绪（正确，未超越 base）** · 优化层面 measurement-bound（host-bound 锁死）

- 产出正确的 naive Triton causal attention（`triton_flexattention_001.py`），harness PASS accuracy，0.612x（比 base 慢，符合预期）
- 优化层面无 ≥5% 可证伪干预：base 是厂商 Ixmma FlashAttention（CausalM_t=2），~91% wall 是 harness-fixed host 开销

## 轮次摘要

| Round | 干预 | 结果 | Wall (ms) | Device (us/call) | Kernel count |
|---|---|---|---|---|---|
| 000 | Phase 0 baseline | baseline | 0.150070 | 12.880 | 0.84 |
| 001 | 无干预（measurement-bound） | aborted | - | - | - |
| - | **参赛交付物**：naive Triton causal attention | **正确（0.612x）** | 0.237728 | - | - |

## 为什么优化层面无空间

### 1. base 是厂商最优的融合 FlashAttention kernel（因果变体）

`base.py` 的 `F.scaled_dot_product_attention(..., is_causal=True)` dispatch 到单 `FlashAttnFwdF16Ixmma<...,CausalM_t=2,AlibiMode_t=0,...>` kernel。这是 task 6（mm_encoder_attention，non-causal，CausalM_t=0）的因果孪生算子，同为 Iluvatar 为 BI150 专门调优的 tensor-core kernel，QK^T + causal softmax + PV 已融合在单 kernel。

### 2. host-bound + harness-fixed

wall = 150 us，device 仅 12.880 us（device_ratio ≈ 0.086）。~91% wall 是 harness 的 `set_seed` + `cuda.synchronize` + 输入 clone，在 candidate 边界外。即使 Triton attention 设备侧完美，也无法达到 5% 阈值（需消除全部 device 时间且零新增 host 成本，而 Triton 重写必然引入自身 launch 开销）。

### 3. tl.dot 已 Supported，但不改变瓶颈

与 task 6 相同的结论：`tl.dot` 在 BI150 上已实测可用（fp32/bf16），但移除 capability-miss 后，瓶颈仍是 host-bound + 厂商 kernel 不可超越。无 (83,83,64) attention tile 的性能证据，`num_warps`/`num_stages` 仍 Unknown。

## 参赛交付物（正确性优先，非优化）

- **candidate**: `triton_flexattention_001.py`（SHA `14c2af71...`）
- **结构**: 单 kernel，grid=(H,)；因果掩码 `offs_s[:,None] >= offs_s[None,:]`（query m, key n, 条件 m >= n）+ valid_key 掩 pad 列；softmax 数值稳定（max 减除，exp(-inf)=0 精确处理掩码列）；scale=0.125，fp16 输入 → fp32 累加
- **验收**: harness `PASS accuracy`（atol=1e-2, rtol=1e-2）
- **速度**: 0.612x（慢于 base 符合预期，base 是厂商 Ixmma FlashAttention）

## 与 task 6 的对比

| | task 6 mm_encoder_attention | task 2 flexattention |
|---|---|---|
| SDPA 类型 | non-causal（CausalM_t=0） | causal（CausalM_t=2） |
| input layout | [2,83,512] batch | [83,8,64] tokens-first |
| 优化结论 | measurement-bound abort | measurement-bound abort（同构） |
| 交付物 | naive Triton attention（0.547x） | naive Triton causal attention（0.612x） |

## 产物

- canonical（优化）: `baseline_adapter.py`（优化层面未产生 accepted candidate）
- 参赛交付物: `triton_flexattention_001.py`
- 决策/报告: `rounds/decision_001.md`（abort）、`rounds/report_000.md`、`rounds/coder_result_deliverable.md`
- 状态: `team-state.md`（`phase: stopped`，`stop_reason: user-intervention`）

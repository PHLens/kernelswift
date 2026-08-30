# mhc_post_layer_mix（BI150）最终总结

## 结论

✅ **完结 · 1.20x**（r001，`8.189047 → 6.427432 ms`）；r002 measurement-bound，stopped

## 轮次摘要

| Round | 干预 | 结果 | Wall (ms) | Device (us/call) | Kernel count |
|---|---|---|---|---|---|
| 000 | Phase 0 baseline | baseline | 8.189047 | 7323.847 | 5.48 |
| 001 | elementwise-fusion（GEMM 后的 cast+乘+加+cast 尾融合为单 kernel） | accepted | 6.427432 | 6122.542 | 2.96 |
| 002 | 无干预（memory-bound 窄 GEMM 无法超越） | aborted | - | - | - |

## 关键证据

- **瓶颈分类**：device-bound（baseline device_ratio 0.894），einsum 走 TCU 批 GEMM（`gemm_tcu_h`，cublasLt）
- **Round 001 收益**：+20.09% wall（`8.043548 → 6.427432 ms`），kernel_count `5.66 → 2.96`，device `7516.836 → 6122.542 us/call`
- **融合内容**：x bf16→fp32 cast + `x*post_layer_mix`（dim=-2 广播）+ `+term2` + fp32→bf16 cast → 单 `_fused_tail_kernel`；GEMM 保持不变
- **正确性**：harness `PASS accuracy`，独立 probe max_abs_diff=0.03125（bf16 精度内），广播与 fp32 中间精度语义验证正确

## 停止边界

r001 后剩余瓶颈是未改的 TCU 批 GEMM（~5183 us/call，占剩余 device ~85%），它是 memory-bound 窄 GEMM（`[4,4]@[4,1280]`，M=4/K=4，每 batch 仅 40960 FLOPs，内存搬运 ~250 MB 主导）。`tl.dot` 的 tile（32×32）与 M=4/K=4 严重不匹配，padding 浪费算力且无法减少内存搬运量，重写预期回退（非能力 miss，是效率不匹配）。cast+tail 已融合且被 cublasLt 库调用隔开，无法进一步融合。r002 Designer 判定无 ≥5% 可证伪干预，stopped。

## 产物

- canonical: `triton_mhc_post_layer_mix_001.py`
- 决策/报告: `rounds/decision_001.md`、`rounds/coder_result_001.md`、`rounds/report_001.md`、`rounds/decision_002.md`
- 状态: `team-state.md`（`phase: stopped`，`stop_reason: user-intervention`）

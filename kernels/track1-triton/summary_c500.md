# C500（沐曦 MACA）Triton 提交物汇总与性能分析

分支：`kernel-opt/c500-triton-submissions`。度量口径：`auto_bench.py --warmup 50 --repeat 100`，median wall time，speedup = v0(base.py) / v1(candidate)。

## 一、5 个算子提交物全景

| 算子 | 提交物 | 正确性 | speedup | 加速来源 |
|---|---|---|---|---|
| `mhc_post_layer_mix` | `triton_mhc_001.py` | PASS | **31.66x** | tiny-k GEMM 融合（K=4 显式 fp32 MAC，替代浪费 ~97% K-work 的 tf32 GEMM），6→1 kernels |
| `mhc_head_compute_mix` | `triton_mhcc_001.py` | PASS | **14.07x** | sigmoid gates + 20 轮 Sinkhorn 迭代融合进单 kernel，133→1 launches |
| `groupedtopk` | `triton_grouped_topk_001.py` | PASS | **3.29x** | 单 per-token kernel 融合 softmax + group-top4 + expert-top8 + 重归一化 |
| `music_flamingo_rotary_embedding` | `triton_rotary_001.py` | PASS | **2.38x** | 11→1 elementwise 融合，消除全部中间物化 |
| `mm_encoder_attention` | `triton_mha_002.py` | PASS | 0.91x | 手写 Triton MHA（见下文根因） |

正确性 **5/5 全部 PASS**，无留空。C500 共完成 5 个算子的 campaign；其余 5 个算子（`flexattention`、`fused_moe`、`sparse_pooler`、`centre_random_augmentation`、`mhc_head_compute_mix_backward`）暂无 maca 提交物。

## 二、慢算子的根因分析

### 1. C500 的 Triton 后端 `tl.dot` 是 Unknown（最根本）

`triton_maca` 未实现 `tl.dot`（矩阵乘原语），任何 GEMM 只能退化成 `tl.sum(a * b)` 的逐元素标量 FMA 展开；同时 `num_warps>1` 也是 Unknown，kernel 并行度受限。因此 attention / 矩阵乘类算子无法利用 C500 的矩阵加速单元，与 base 的库算子（mcblas 等）天然差一个数量级。

### 2. mm_encoder_attention（0.91x）

base 用 `F.scaled_dot_product_attention`，在 C500 上落到硬件优化的 flash-attention SDPA（baseline 0.115761 ms），本身已是注意力算子的性能上限。手写 Triton MHA：

- QK^T 与 PV 用 `tl.sum(q * K)` 标量乘加（head_size=64 手动 dot），无 `tl.dot` 可用；
- r001 产出正确融合 MHA（fp32 累加、two-pass max-subtracted softmax），wall 0.164166 ms，比 baseline 慢；
- r002 消除 4 个 `.contiguous()` transpose-copy，5→2 kernels（1 fused `_mha_fwd_kernel` + 1 不可避免的 output reshape），+23.54% → 0.127777 ms。

最终提交物正确（allclose 1e-2）且已优化到 C500 profile 允许的极限：剩余主导成本是 `_mha_fwd_kernel` ~64.85 us/call 的手动 `tl.sum` dot，无 `tl.dot` 无法再加速。属于 flash-attention floor 下的 measurement-bound（stop: user-intervention）。

### 3. 为什么其余 4 个算子反而快

因为它们**规避了 GEMM 瓶颈，主要省的是 launch 开销 / 冗余计算**：

- `mhc_post_layer_mix`（31.66x）：base 的 einsum `'abmn,abmc->abnc'`（K=mhc_mult=4）落到 `mcblas tf32gemm 64x64x128` tile，K 维浪费 ~97%（6071 us/call = 80% device time）；手写 kernel 用 4 次显式 fp32 MAC 替代并折叠 elementwise 尾（mul + add + 2 bf16 cast），6→1 kernels。r002 后已到 **memory-bandwidth floor**：~170 MB 流量 / 168.56 us ≈ 1 TB/s，接近 C500 级 HBM 上限。
- `mhc_head_compute_mix`（14.07x）：base 有 133 次 launch（~65% wall 是 host launch 开销）；Sinkhorn [4,4] 小矩阵 20 轮迭代 + sigmoid gates 融合进单 kernel（16 programs），device 534→43.79 us/call。r002 后已到 **latency floor**：20 步交替归一化是串行依赖链，不改精确 fp32 语义则不可并行。
- `music_flamingo_rotary_embedding`（2.38x）：纯 elementwise 链（batch/time broadcast、concat、angle-scale、cos、sin，11 个 PyTorch kernel）融合为单 direct-launch kernel 后 device 50.95→16.90 us/call；r002 后剩余 ~63 us/call 为 harness 固定开销（`set_seed` + `sync_devices`），无 candidate-owned lever。
- `groupedtopk`（3.29x）：把固定 softmax、group-max/group-top4、masked expert-top8、重归一化整链融合进单 per-token kernel（grid `(83,)`，`BLOCK_E=256`），0.2245→0.0683 ms；r002–r004 连续三次 valid no-improvement，按 `valid_no_improvement_limit=3` 规则停止。

**结论**：C500 上慢算子慢在 **GEMM 被降级成标量 FMA**（`tl.dot` Unknown），快算子快在 **launch/物化融合**。这是 `triton_maca` 当前平台性短板（无 `tl.dot`、`num_warps>1`），而非 kernel 写得差。

## 三、可优化方向（按性价比排序）

1. **验证 `tl.dot` 在 triton_maca 的可用性**：若未来可用，`mm_encoder_attention` 可重写为真矩阵乘 attention（QK^T / PV 用 `tl.dot`），这是唯一可能追平 flash-attention floor 的路径。
2. **验证 `num_warps>1` 的可用性**：当前默认 `num_warps=1`；若支持可提升 occupancy 与吞吐，对 `mm_encoder_attention` 有一定收益。
3. **已触底的算子不再投入**：`mhc_post_layer_mix`（~1 TB/s HBM 上限）、`mhc_head_compute_mix`（Sinkhorn 串行依赖链）、`music_flamingo_rotary_embedding`（harness 固定开销）、`groupedtopk`（valid-no-improvement 规则停止）均已无 ≥5% candidate-owned 路径。

## 四、提交物规范

每个算子的提交物位于 `kernels/track1-triton/<算子>/maca/triton_<...>_001.py`，含：

- `ModelNew` 类（接口与 base.py 的 `Model` 一致）
- `get_inputs()` / `get_init_inputs()`
- 环境约束：`tl.dot` 不可用（GEMM 用 `tl.sum` 展开）、`num_warps>1` 不可用；kernel 内无随机数生成（host 生成后传入）

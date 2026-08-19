# Ascend 910B（昇腾）Triton 提交物汇总与性能分析

分支：`kernel-opt/ascend910b-20260818-053937`（campaign 工作区，经 PR #20 导入）。
度量口径：`auto_bench.py --warmup 50 --repeat 100`，median wall time，speedup = v0(base.py) / v1(candidate)。
环境：`Ascend910B4`，`Triton-Ascend 3.2.1`（triton 3.2.0），`CANN 9.0.0`，`torch_npu 2.7.1.post4`，Python 3.11。
设备时长来自 CANN msprof `ai_core_op_summary.db`（`torch_npu.profiler` 每 scope 独立采集，`summarize_cann_trace.py` 归一化）。

## 一、10 个算子提交物全景

| 算子 | 提交物 | 正确性 | speedup | 加速来源 |
|---|---|---|---|---|
| `groupedtopk` | `triton_grouped_topk_002.py` | PASS | **2.84x** | 单 per-token kernel 融合（12→1 launch）+ 输出 buffer 复用 |
| `fused_moe` | `triton_fused_moe_003.py` | PASS | **19.2x** | 逐-token 路由 + per-expert GEMM 融合 + host 输出缓存 |
| `mhc_head_compute_mix` | `candidate_001.py` | PASS | **8.99x** | Sinkhorn 20 轮迭代融合进单 kernel（wall 3.527→0.392 ms），消除 20×多次 launch |
| `mhc_post_layer_mix` | `candidate_001.py` | PASS | **1.00x**（≈打平） | einsum 融合（wall 3.212→3.198 ms）；r002 报告 0.886 ms 但相对口径不一致且未推进 canonical |
| `music_flamingo_rotary_embedding` | `triton_rotary_001.py` | PASS | **1.75x** | 逐-token cos/sin 计算融合（0.581→0.334 ms） |
| `flexattention` | `triton_flexattention_002.py` | PASS | **1.45x** | causal SDPA 单 kernel 融合（0.409→0.282 ms） |
| `sparse_pooler` | `triton_sparse_pooler_001.py` | PASS | **1.51x** | log1p+ReLU+max-pool 融合（0.936→0.619 ms） |
| `centre_random_augmentation` | `triton_centre_random_aug_001.py` | PASS | **1.03x** | 四元数旋转矩阵融合（2.548→2.463 ms，+17.84% wall 但受 host 随机数生成限制） |
| `mhc_head_compute_mix_backward` | `triton_mhc_mix_bwd_001.py` | PASS | **1.02x** | elementwise sigmoid-backward 融合（0.457→0.446 ms，+3.26% 未达 5% 阈值） |
| `mm_encoder_attention` | `triton_attn_001.py` | PASS | **0.92x** | 手写 Triton attention（0.321→0.349 ms，+2.56% 未达 5%；原生 FA 已接近最优） |

正确性 **10/10 全部 PASS**，无留空。

## 二、性能分析

### 1. 整体格局：绝大多数算子由 host 开销主导

Ascend 910B 的 Triton 后端（triton_ascend）对 `tl.dot`、`fast_libentry`、`async_copy` 等原语仍属
`Unknown`（见 `skills/kernel-opt-loop/prompts/coder_targets/triton_ascend.md`），但直接 Triton launch
是成熟路径。在小 shape（T=83 量级）下，device 时间只占 wall 的一小部分，wall 主要由 host 侧
launch/dispatch 开销决定：

| 算子 | wall (ms) | device us/call | device_ratio | 停止原因 |
|---|---:|---:|---:|---|
| `groupedtopk` r002 | 0.267 | 35.13 | 0.131 | host-bound-remaining-cost-fixed |
| `flexattention` r002 | 0.282 | 54.64 | ~0.19 | host-bound-remaining-cost-fixed |
| `fused_moe` r003 | 0.373 | 26.62 | ~0.07 | host-bound-remaining-cost-fixed |
| `sparse_pooler` r001 | 0.619 | 202.86 | ~0.33 | no-falsifiable-intervention-remains |

### 2. 慢算子的根因分析

- **`mm_encoder_attention`（0.92x）/ `flexattention` 后续轮（r003 -8.34%）**：手写 Triton SDPA
  的 QK^T/PV 无法利用矩阵单元（`tl.dot` Unknown），且每 query 一个 program、`num_warps=1`，K/V
  无跨 program 复用。base 的 `F.scaled_dot_product_attention` 落到原生 FlashAttention 融合算子，
  手写版本只能在小 shape 下靠减少 launch 与库算子打平。
- **`centre_random_augmentation`（1.03x）**：融合后 wall 2.548→2.463 ms（+17.84% 相对 base），
  但随机数生成在 host 侧，host-bound-floor，后续无可证伪干预。
- **`mhc_head_compute_mix` 的 Sinkhorn 融合收益**：device 282.354→8.784 us/call（约 32x），
  wall 3.527→0.392 ms 的主收益来自 20 轮迭代从 20×多次 launch 融合为单 kernel（per-launch
  host 开销塌缩）。
- **`mhc_post_layer_mix`（wall ≈ 打平）**：r001 单 kernel 融合后 device 从 3082.85 压到 619.76
  us/call（+264% device 口径），wall 3.212→3.198 ms 未达 5% 阈值；r002 相对 accepted
  `candidate_001` 为 -0.58%（no-improvement），canonical 未推进，判定无可证伪干预。
- **`mhc_head_compute_mix_backward`（1.02x）**：正确性 PASS 但 wall 仅 +3.26%，低于 5% adoption
  阈值，交付为 Triton 提交物并停止。

### 3. 为什么 groupedtopk / fused_moe / mhc_head_compute_mix / sparse_pooler 快

它们**规避了 GEMM/attention 瓶颈**，收益主要来自 launch 消除与 host 开销压缩：

- `groupedtopk`：r001 单 kernel 融合 +54.88%，r002 输出 buffer 复用 +18.21%（host 291.7→232.1
  us/call）；device 稳定 ~35 us，剩余 host 为 Triton launch/dispatch 固定开销（~107 us 独立于
  kernel 大小），无法在 kernel 侧压缩。
- `fused_moe`：r001 逐-token 路由融合 +92.71%，r002 输出缓存 +35.94%，r003 host 路径优化
  +6.70%（device 稳定 ~27 us）。
- `mhc_head_compute_mix`：20 轮 Sinkhorn 迭代从多次 launch 融合为单 kernel，wall 3.527→0.392 ms
  （8.99x），收益全部来自 host launch 消除。
- `sparse_pooler`：r001 融合 +33.78%（wall 0.936→0.619 ms）；r002 再融合仅 +2.75%，低于阈值。

## 三、可优化方向（按性价比排序）

1. **验证 `tl.dot` / `fast_libentry` 在 triton_ascend 的可用性**：若 910B 的 Triton 后端支持
   `tl.dot`（矩阵单元）或提供 fast-launcher 路径，attention/GEMM 类算子（mm_encoder_attention、
   flexattention、mhc_post_layer_mix）有明确的 device/host 双收益；当前 profile 将这些原语列为
   `Unknown`，需要先做匹配的本地 probe（见 `triton_ascend.md` 的 Unknown 表）。
2. **压缩 host launch 开销**：对小 shape 算子，host dispatch（~107-232 us/call）是 wall 的最大
   单点。若 `fast_libentry` 或等价 launcher 路径在 910B 上可用，groupedtopk/flexattention 等
   host-bound 算子可继续获得 5%+ 收益；否则剩余 host 成本是后端固定开销。
3. **增大 tile / num_warps**：手写 SDPA 当前每 query 一个 program、`num_warps=1`，可尝试一个
   program 处理多个 query 提升 occupancy；但追平库算子仍需矩阵单元支持。

## 四、提交物规范

每个算子的提交物位于 `kernels/track1-triton/<算子>/ascend/<提交物>.py`，含：

- `ModelNew` 类（接口与 `base.py` 的 `Model` 一致），共享设备无关 `base.py`（`../base.py`）
- `get_inputs()` / `get_init_inputs()`（harness 自动把 `cuda` 占位改写为 `npu`）
- campaign 完整证据链：`project.md`（runtime/measurement fingerprint）、`team-state.md`、
  `rounds/`（decision/coder_result/report/round_status）、`state/`
- profiler：`auto_bench.py --profile` 在 NPU 上经 `torch_npu.profiler` + CANN msprof 分 scope
  采集，`skills/kernel-opt-loop/scripts/summarize_cann_trace.py` 归一化

## 五、环境备注

- NPU profiler 的 CANN sqlite 无 scope 字段且时钟与 chrome trace 不一致，故 harness 对每个
  scope（reference/candidate）分别采集一次，每个 `ASCEND_WORK_PATH` 子目录对应一个 scope。
- 所有 campaign 均已停止（host-bound / measurement-bound / no-falsifiable-intervention-remains），
  无正在运行的轮次。

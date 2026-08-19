# 赛道一：算子 × 后端 进度矩阵

度量口径：`auto_bench.py`（`--warmup 50 --repeat 100`，median wall）。
加速比为相对各后端共享 `base.py` 的累计结果；"运行中" 项在 run 分支/worktree 上。
task 编号 ↔ 算子目录映射见 [docs/competition/track1-triton.md](../../docs/competition/track1-triton.md)。

| 算子 | `mlu`（寒武纪 MLU590） | `s60`（燧原 GCU） | `maca`（沐曦 C500） | `bi150`（天数智芯） | `ascend910b`（昇腾） |
|---|---|---|---|---|---|
| `groupedtopk` | ✅ 完结 · **6.56x**（v4，0.840→0.128 ms） | ✅ 完结 · **1.68x**（r003，0.459→0.274 ms）；r004 +2.06% 未达 5% 阈值 | ✅ 完结 · **3.29x**（r001，0.225→0.068 ms）；r002–004 未达 5% 阈值，stopped | ✅ 完结 · **1.71x**（r009，0.475→0.277 ms） | ✅ 完结 · **2.84x**（r002，0.760→0.267 ms） |
| `flexattention` | ✅ 完结 · **7.08x**（v3，1.006→0.140 ms） | ⛔ 无优化空间 · r000 baseline（0.269 ms）已单 kernel 融合，分支 `kernel-opt/flexattention-s60` | — | — | ✅ 完结 · **1.45x**（r002，0.409→0.282 ms） |
| `fused_moe` | ✅ 完结 · **50.4x**（v5，6.94→0.138 ms） | — | — | — | ✅ 完结 · **19.2x**（r003，7.16→0.373 ms） |
| `sparse_pooler` | ✅ 完结 · **1.60x**（v4，0.910→0.567 ms） | ⛔ 无优化空间 · r001 融合 -26.79%（手写 Triton 慢于库算子），分支 `kernel-opt/sparse-pooler-s60` | — | — | ✅ 完结 · **1.51x**（r001，0.936→0.619 ms） |
| `music_flamingo_rotary_embedding` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **1.86x**（r001，0.622→0.334 ms） |
| `mm_encoder_attention` | 📦 base.py 就绪（待 Phase 0） | — | — | — | 🟡 已交付 Triton 版本 · r001 +2.56%（0.349→0.340 ms），未达 5% 阈值 |
| `mhc_post_layer_mix` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **3.64x**（r001，3.198→0.880 ms） |
| `mhc_head_compute_mix` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **9.00x**（r001，3.527→0.392 ms） |
| `centre_random_augmentation` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **1.22x**（r001，2.463→2.024 ms） |
| `mhc_head_compute_mix_backward` | 📦 base.py 就绪（待 Phase 0） | — | — | — | 🟡 已交付 Triton 版本 · r001 +3.26%（0.446→0.431 ms），未达 5% 阈值 |

## Ascend910B 导入说明

PR #20 导入了 10 个 `ascend/` campaign workspace；这些 campaign 当前都已停止迭代（stopped）。
其中：

- `✅` 表示已有 accepted canonical，可直接把 wall-time 结果记入矩阵；
- `🟡` 表示已交付 Triton 候选且 correctness 通过，但 wall 提升未跨过 `5%` adoption threshold。

## groupedtopk：S60 与 MLU 的差异

S60 campaign 已在 r003 固化 canonical（`0.459285 -> 0.273673 ms`，`1.68x`），但不能把 MLU 的 `6.56x` 结果直接迁移或承诺为 S60 的可达结果：

| 维度 | MLU590 | S60 GCU |
|---|---|---|
| 已接受 canonical | v4，`0.128 ms` | r003，`0.273673 ms` |
| 已验证主要收益 | 单 Triton fusion、输出复用、`tl.argmax` 将 selection-sort reduction 从 3 轮降到 2 轮、host constexpr 缓存 | 单 Triton-GCU launch（`12 -> 1/call`）、输出池复用、exact-key metadata cache |
| 后续 device 方向 | profiler 可测 device time；v3 将 device `38.1 -> 20.0 us`，但 v4 后 wall 已由 host 固定成本主导 | `torch_gcu`/TOPSPTI 可识别 `_grouped_topk_kernel` 和 launch metadata，但当前记录的 kernel `start/end` 为零，无法得到可用 device duration |
| 停止边界 | v4 后 kernel-side 增益不能再显著改善 wall | r004 的 host stream/context specialization 已正确但仅 `+2.06%`，低于 `5%` adoption threshold；r005-r008 没有可证实的 >=5% candidate path |

这不是 GCU 硬件或 Triton-GCU 必然无法达到 MLU 数值，而是当前 S60 runtime/profile 无法为 MLU 的 selection/dataflow 路径提供等价的 device-time、lowering 和 tie-semantics 证据。后续须先取得与该 runtime 匹配的 TopsProf/TOPSPTI device-duration exporter，或同运行时 microbenchmark 证明 candidate-owned bottleneck，才应重新开启该方向。

## 各后端 campaign 目录

| 算子 | base | campaign 根 | 结论 |
|---|---|---|---|
| `groupedtopk` | [base.py](groupedtopk/base.py)（设备无关共享） | [mlu](groupedtopk/mlu/) · [s60](groupedtopk/s60/) · [maca](groupedtopk/maca/) · [bi150](groupedtopk/bi150/) · [ascend](groupedtopk/ascend/) | [mlu outcome](groupedtopk/mlu/outcome.md) · [s60 final summary](groupedtopk/s60/final_summary.md) · [maca final summary](groupedtopk/maca/final_summary.md) · [bi150 r009 report](groupedtopk/bi150/rounds/report_009.md) · [ascend r002 report](groupedtopk/ascend/rounds/report_002.md) |
| `flexattention` | [base.py](flexattention/base.py)（设备无关共享） | [mlu](flexattention/mlu/) · [s60](flexattention/s60/) · [ascend](flexattention/ascend/) | [mlu outcome](flexattention/mlu/outcome.md) · [s60 project](flexattention/s60/project.md) · [ascend r002 report](flexattention/ascend/rounds/report_002.md) |
| `fused_moe` | [base.py](fused_moe/base.py) | [mlu](fused_moe/mlu/) · [ascend](fused_moe/ascend/) | [mlu outcome](fused_moe/mlu/outcome.md)（bangc 探针见 track2） · [ascend r003 report](fused_moe/ascend/rounds/report_003.md) |
| `sparse_pooler` | [base.py](sparse_pooler/base.py)（设备无关共享） | [mlu](sparse_pooler/mlu/) · [s60](sparse_pooler/s60/) · [ascend](sparse_pooler/ascend/) | [mlu project](sparse_pooler/mlu/project.md) · [s60 project](sparse_pooler/s60/project.md) · [ascend r001 report](sparse_pooler/ascend/rounds/report_001.md) |
| `music_flamingo_rotary_embedding` | [base.py](music_flamingo_rotary_embedding/base.py) | [ascend](music_flamingo_rotary_embedding/ascend/) | [ascend r001 report](music_flamingo_rotary_embedding/ascend/rounds/report_001.md) |
| `mm_encoder_attention` | [base.py](mm_encoder_attention/base.py) | [ascend](mm_encoder_attention/ascend/) | [ascend r001 report](mm_encoder_attention/ascend/rounds/report_001.md) |
| `mhc_post_layer_mix` | [base.py](mhc_post_layer_mix/base.py) | [ascend](mhc_post_layer_mix/ascend/) | [ascend r001 report](mhc_post_layer_mix/ascend/rounds/report_001.md) |
| `mhc_head_compute_mix` | [base.py](mhc_head_compute_mix/base.py) | [ascend](mhc_head_compute_mix/ascend/) | [ascend r001 report](mhc_head_compute_mix/ascend/rounds/report_001.md) |
| `centre_random_augmentation` | [base.py](centre_random_augmentation/base.py) | [ascend](centre_random_augmentation/ascend/) | [ascend r001 report](centre_random_augmentation/ascend/rounds/report_001.md) |
| `mhc_head_compute_mix_backward` | [base.py](mhc_head_compute_mix_backward/base.py) | [ascend](mhc_head_compute_mix_backward/ascend/) | [ascend r001 report](mhc_head_compute_mix_backward/ascend/rounds/report_001.md) |

## 维护

新增/完结 campaign 时更新本表。可考虑用 `scripts/update_matrix.py` 从各
`project.md` / `outcome.md` 自动汇总（尚未实现）。

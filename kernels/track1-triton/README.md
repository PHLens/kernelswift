# 赛道一：算子 × 后端 进度矩阵

度量口径：`auto_bench.py`（`--warmup 50 --repeat 100`，median wall）。
加速比为相对各后端共享 `base.py` 的累计结果；"运行中" 项在 run 分支/worktree 上。
task 编号 ↔ 算子目录映射见 [docs/competition/track1-triton.md](../../docs/competition/track1-triton.md)。

| 算子 | `mlu`（寒武纪 MLU590） | `s60`（燧原 GCU） | `maca`（沐曦 C500） | `bi150`（天数智芯） | `ascend910b`（昇腾） |
|---|---|---|---|---|---|
| `groupedtopk` | ✅ 完结 · **6.56x**（v4，0.840→0.128 ms） | ✅ 完结 · **1.68x**（r003，0.459→0.274 ms）；r004 +2.06% 未达 5% 阈值 | ✅ 完结 · **3.29x**（r001，0.225→0.068 ms）；r002–004 未达 5% 阈值，stopped | 🔄 运行中 · 分支 `kernel-opt/bi150-prepare-20260818` | — |
| `flexattention` | ✅ 完结 · **7.08x**（v3，1.006→0.140 ms） | ⛔ 无优化空间 · r000 baseline（0.269 ms）已单 kernel 融合，分支 `kernel-opt/flexattention-s60` | — | — | — |
| `fused_moe` | ✅ 完结 · **50.4x**（v5，6.94→0.138 ms） | 🔄 运行中 · r001 **10.55x**（5.26→0.499 ms，147→8 launches），分支 `kernel-opt/fused-moe-s60` | — | — | — |
| `sparse_pooler` | ✅ 完结 · **1.60x**（v4，0.910→0.567 ms） | — | — | — | — |
| `music_flamingo_rotary_embedding` | 📦 base.py 就绪（待 Phase 0） | — | — | — | — |
| `mm_encoder_attention` | 📦 base.py 就绪（待 Phase 0） | — | — | — | — |
| `mhc_post_layer_mix` | 📦 base.py 就绪（待 Phase 0） | — | — | — | — |
| `mhc_head_compute_mix` | 📦 base.py 就绪（待 Phase 0） | — | — | — | — |
| `centre_random_augmentation` | 📦 base.py 就绪（待 Phase 0） | — | — | — | — |
| `mhc_head_compute_mix_backward` | 📦 base.py 就绪（待 Phase 0） | — | — | — | — |

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
| `groupedtopk` | [base.py](groupedtopk/base.py)（设备无关共享） | [mlu](groupedtopk/mlu/) · [s60](groupedtopk/s60/) · [maca](groupedtopk/maca/) | [mlu outcome](groupedtopk/mlu/outcome.md) · [s60 final summary](groupedtopk/s60/final_summary.md) · [maca final summary](groupedtopk/maca/final_summary.md) |
| `flexattention` | [base.py](flexattention/base.py)（设备无关共享） | [mlu](flexattention/mlu/) · [s60](flexattention/s60/) | [mlu outcome](flexattention/mlu/outcome.md) · [s60 project](flexattention/s60/project.md) |
| `fused_moe` | [base.py](fused_moe/base.py)（设备无关共享） | [mlu](fused_moe/mlu/) · [s60](fused_moe/s60/) | [mlu outcome](fused_moe/mlu/outcome.md) · [s60 project](fused_moe/s60/project.md) |
| `sparse_pooler` | [base.py](sparse_pooler/base.py) | [mlu](sparse_pooler/mlu/) | [project](sparse_pooler/mlu/project.md) |
| `music_flamingo_rotary_embedding` | [base.py](music_flamingo_rotary_embedding/base.py) | 待 Phase 0 | — |
| `mm_encoder_attention` | [base.py](mm_encoder_attention/base.py) | 待 Phase 0 | — |
| `mhc_post_layer_mix` | [base.py](mhc_post_layer_mix/base.py) | 待 Phase 0 | — |
| `mhc_head_compute_mix` | [base.py](mhc_head_compute_mix/base.py) | 待 Phase 0 | — |
| `centre_random_augmentation` | [base.py](centre_random_augmentation/base.py) | 待 Phase 0 | — |
| `mhc_head_compute_mix_backward` | [base.py](mhc_head_compute_mix_backward/base.py) | 待 Phase 0 | — |

## 维护

新增/完结 campaign 时更新本表。可考虑用 `scripts/update_matrix.py` 从各
`project.md` / `outcome.md` 自动汇总（尚未实现）。

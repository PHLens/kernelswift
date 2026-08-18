# 赛道一：算子 × 后端 进度矩阵

度量口径：`auto_bench.py`（`--warmup 50 --repeat 100`，median wall）。
加速比为相对各后端共享 `base.py` 的累计结果；"运行中" 项在 run 分支/worktree 上。
task 编号 ↔ 算子目录映射见 [docs/competition/track1-triton.md](../../docs/competition/track1-triton.md)。

| 算子 | `mlu`（寒武纪 MLU590） | `s60`（燧原 GCU） | `maca`（沐曦 C500） | `bi150`（天数智芯） | `ascend910b`（昇腾） |
|---|---|---|---|---|---|
| `groupedtopk` | ✅ 完结 · **6.56x**（v4，0.840→0.128 ms） | 🔄 运行中 · r001 +39.1%（0.459→0.274 ms），分支 `kernel-opt/groupedtopk-s60*` | 🔄 运行中 · 分支 `kernel-opt/grouptopk-c500-20260818` | 🔄 运行中 · 分支 `kernel-opt/bi150-prepare-20260818` | ✅ 完结 · **2.84x**（v2，0.760→0.267 ms） |
| `flexattention` | ✅ 完结 · **7.08x**（v3，1.006→0.140 ms） | — | — | — | ✅ 完结 · **1.45x**（v2，0.409→0.282 ms） |
| `fused_moe` | ✅ 完结 · **50.4x**（v5，6.94→0.138 ms） | — | — | — | ✅ 完结 · **19.2x**（v3，7.16→0.373 ms） |
| `sparse_pooler` | ✅ 完结 · **1.60x**（v4，0.910→0.567 ms） | — | — | — | ✅ 完结 · **1.51x**（v1，0.936→0.619 ms） |
| `music_flamingo_rotary_embedding` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **1.46x**（v1，0.622→0.334 ms） |
| `mm_encoder_attention` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **1.03x**（v1，0.349→0.340 ms；原生 FA 已最优） |
| `mhc_post_layer_mix` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **3.64x**（v1，3.212→0.880 ms） |
| `mhc_head_compute_mix` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **8.99x**（v1，3.527→0.392 ms） |
| `centre_random_augmentation` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **1.22x**（v1，2.463→2.024 ms） |
| `mhc_head_compute_mix_backward` | 📦 base.py 就绪（待 Phase 0） | — | — | — | ✅ 完结 · **1.03x**（v1，0.446→0.431 ms） |

## 各后端 campaign 目录

| 算子 | base | campaign 根 | 结论 |
|---|---|---|---|
| `groupedtopk` | [base.py](groupedtopk/base.py)（设备无关共享） | [mlu](groupedtopk/mlu/) · [s60](groupedtopk/s60/) · [ascend](groupedtopk/ascend/) | [mlu outcome](groupedtopk/mlu/outcome.md) · [s60 project](groupedtopk/s60/project.md) · [ascend project](groupedtopk/ascend/project.md) |
| `flexattention` | [base.py](flexattention/base.py) | [mlu](flexattention/mlu/) · [ascend](flexattention/ascend/) | [mlu outcome](flexattention/mlu/outcome.md) · [ascend project](flexattention/ascend/project.md) |
| `fused_moe` | [base.py](fused_moe/base.py) | [mlu](fused_moe/mlu/) · [ascend](fused_moe/ascend/) | [mlu outcome](fused_moe/mlu/outcome.md)（bangc 探针见 track2） · [ascend project](fused_moe/ascend/project.md) |
| `sparse_pooler` | [base.py](sparse_pooler/base.py) | [mlu](sparse_pooler/mlu/) · [ascend](sparse_pooler/ascend/) | [mlu project](sparse_pooler/mlu/project.md) · [ascend project](sparse_pooler/ascend/project.md) |
| `music_flamingo_rotary_embedding` | [base.py](music_flamingo_rotary_embedding/base.py) | [ascend](music_flamingo_rotary_embedding/ascend/) | [ascend project](music_flamingo_rotary_embedding/ascend/project.md) |
| `mm_encoder_attention` | [base.py](mm_encoder_attention/base.py) | [ascend](mm_encoder_attention/ascend/) | [ascend project](mm_encoder_attention/ascend/project.md) |
| `mhc_post_layer_mix` | [base.py](mhc_post_layer_mix/base.py) | [ascend](mhc_post_layer_mix/ascend/) | [ascend project](mhc_post_layer_mix/ascend/project.md) |
| `mhc_head_compute_mix` | [base.py](mhc_head_compute_mix/base.py) | [ascend](mhc_head_compute_mix/ascend/) | [ascend project](mhc_head_compute_mix/ascend/project.md) |
| `centre_random_augmentation` | [base.py](centre_random_augmentation/base.py) | [ascend](centre_random_augmentation/ascend/) | [ascend project](centre_random_augmentation/ascend/project.md) |
| `mhc_head_compute_mix_backward` | [base.py](mhc_head_compute_mix_backward/base.py) | [ascend](mhc_head_compute_mix_backward/ascend/) | [ascend project](mhc_head_compute_mix_backward/ascend/project.md) |

## 维护

新增/完结 campaign 时更新本表。可考虑用 `scripts/update_matrix.py` 从各
`project.md` / `outcome.md` 自动汇总（尚未实现）。

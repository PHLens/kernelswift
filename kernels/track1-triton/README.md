# 赛道一：算子 × 后端 进度矩阵

度量口径：`auto_bench.py`（`--warmup 50 --repeat 100`，median wall）。
表格单元格式：状态 · 加速比 · 最优轮次 · campaign baseline→最佳 candidate wall ms。加速比为累计值；若最佳轮次相对上一 accepted candidate 测量，则使用对应 baseline 与最佳 candidate 的现有 artifacts 计算。
task 编号 ↔ 算子目录映射见 [docs/competition/track1-triton.md](../../docs/competition/track1-triton.md)。

| 算子 | `mlu`（寒武纪 MLU590） | `s60`（燧原 GCU） | `maca`（沐曦 C500） | `bi150`（天数智芯） | `ascend910b`（昇腾） |
|---|---|---|---|---|---|
| `groupedtopk` | ✅ **6.56x** · v4 · **0.840→0.128 ms** | ✅ **1.68x** · r003 · **0.459→0.274 ms** | ✅ **3.29x** · r001 · **0.225→0.068 ms** | ✅ **1.71x** · r009 · **0.475→0.277 ms** | ✅ **2.84x** · r002 · **0.760→0.267 ms** |
| `flexattention` | ✅ **7.08x** · v3 · **1.006→0.140 ms** | 🟡 **0.42x** · r001 · **0.269→0.64 ms**（correctness PASS，手写 causal SDPA，慢因 `tl.dot` 缺失） | — | 🟡 **0.61x** · r001 · **0.150→0.238 ms** | ✅ **1.45x** · r002 · **0.409→0.282 ms** |
| `fused_moe` | ✅ **50.4x** · v5 · **6.940→0.138 ms** | ✅ **13.1x** · r002 · **5.112→0.390 ms**（逐-token 路由 + selection 融合） | — | ✅ **6.60x** · r002 · **3.259→0.493 ms** | ✅ **19.4x** · r002 · **7.159→0.369 ms** |
| `sparse_pooler` | ✅ **1.60x** · v4 · **0.910→0.567 ms** | 🟡 **0.79x** · r001 · **0.861→1.092 ms** | — | ✅ **1.22x** · r001 · **1.070→0.880 ms** | ✅ **1.51x** · r001 · **0.935→0.619 ms** |
| `music_flamingo_rotary_embedding` | 📦 — · — · — | 🟡 **0.9x** · r002（elementwise 融合，measurement-bound） | ✅ **2.38x** · r001 · **0.191→0.080 ms** | ✅ **1.95x** · r001 · **0.353→0.176 ms** | ✅ **1.86x** · r001 · **0.622→0.334 ms** |
| `mm_encoder_attention` | 📦 — · — · — | 🟡 **0.27x** · r001（手写 SDPA，慢因 `tl.dot` 缺失） | 🟡 **0.91x** · r002 · **0.116→0.128 ms**（手写 Triton MHA，flash-attn 已最优） | 🟡 **0.55x** · r001 · **0.151→0.239 ms** | 🟡 **1.03x** · r001 · **0.349→0.340 ms** |
| `mhc_post_layer_mix` | 📦 — · — · — | 🟡 **0.56x** · r001（einsum 用 `tl.sum` 展开） | ✅ **31.66x** · r001 · **7.636→0.241 ms** | ✅ **1.20x** · r001 · **8.189→6.427 ms** | ✅ **3.64x** · r001 · **3.198→0.880 ms** |
| `mhc_head_compute_mix` | 📦 — · — · — | ✅ **6.8x** · r001（Sinkhorn 迭代融合） | ✅ **14.07x** · r001 · **1.515→0.118 ms** | ✅ **7.79x** · r001 · **1.433→0.184 ms** | ✅ **9.00x** · r001 · **3.527→0.392 ms** |
| `centre_random_augmentation` | 📦 — · — · — | 🟡 **0.95x** · r001（四元数旋转） | — | ✅ **4.49x** · r002 · **1.073→0.239 ms** | ✅ **1.22x** · r001 · **2.463→2.024 ms** |
| `mhc_head_compute_mix_backward` | 📦 — · — · — | 🟡 **1.26x** · r001（sigmoid-backward 融合） | — | ✅ **1.76x** · r001 · **0.351→0.199 ms** | 🟡 **1.03x** · r001 · **0.446→0.431 ms** |

## 表项说明

- `✅` correctness 通过，已提交 Triton code，wall speedup 达到 `5%` threshold；
- `🟡` correctness 通过，已提交 Triton code，但 wall speedup 未达到 `5%` threshold，仍保留实测加速比；
- `⛔` 没有可接受的 Triton candidate；`—` 表示没有对应 campaign artifact 或没有可测 candidate；
- `📦` 仅有 `base.py`，尚无该后端的 Triton submission。

## 各后端 campaign 目录

| 算子 | base | campaign 根 | 结论 |
|---|---|---|---|
| `groupedtopk` | [base.py](groupedtopk/base.py)（设备无关共享） | [mlu](groupedtopk/mlu/) · [s60](groupedtopk/s60/) · [maca](groupedtopk/maca/) · [bi150](groupedtopk/bi150/) · [ascend](groupedtopk/ascend/) | [mlu outcome](groupedtopk/mlu/outcome.md) · [s60 final summary](groupedtopk/s60/final_summary.md) · [maca final summary](groupedtopk/maca/final_summary.md) · [bi150 r009 report](groupedtopk/bi150/rounds/report_009.md) · [ascend r002 report](groupedtopk/ascend/rounds/report_002.md) |
| `flexattention` | [base.py](flexattention/base.py)（设备无关共享） | [mlu](flexattention/mlu/) · [s60](flexattention/s60/) · [bi150](flexattention/bi150/) · [ascend](flexattention/ascend/) | [mlu outcome](flexattention/mlu/outcome.md) · [s60 project](flexattention/s60/project.md) · [bi150 final summary](flexattention/bi150/final_summary.md) · [ascend r002 report](flexattention/ascend/rounds/report_002.md) |
| `fused_moe` | [base.py](fused_moe/base.py)（设备无关共享） | [mlu](fused_moe/mlu/) · [s60](fused_moe/s60/) · [bi150](fused_moe/bi150/) · [ascend](fused_moe/ascend/) | [mlu outcome](fused_moe/mlu/outcome.md)（bangc 探针见 track2） · [s60 project](fused_moe/s60/project.md) · [bi150 final summary](fused_moe/bi150/final_summary.md) · [ascend r002 report](fused_moe/ascend/rounds/report_002.md) |
| `sparse_pooler` | [base.py](sparse_pooler/base.py)（设备无关共享） | [mlu](sparse_pooler/mlu/) · [s60](sparse_pooler/s60/) · [bi150](sparse_pooler/bi150/) · [ascend](sparse_pooler/ascend/) | [mlu project](sparse_pooler/mlu/project.md) · [s60 project](sparse_pooler/s60/project.md) · [bi150 final summary](sparse_pooler/bi150/final_summary.md) · [ascend r001 report](sparse_pooler/ascend/rounds/report_001.md) |
| `music_flamingo_rotary_embedding` | [base.py](music_flamingo_rotary_embedding/base.py) | [s60](music_flamingo_rotary_embedding/s60/) · [bi150](music_flamingo_rotary_embedding/bi150/) · [ascend](music_flamingo_rotary_embedding/ascend/) · [maca](music_flamingo_rotary_embedding/maca/) | [s60 project](music_flamingo_rotary_embedding/s60/project.md) · [bi150 final summary](music_flamingo_rotary_embedding/bi150/final_summary.md) · [ascend r001 report](music_flamingo_rotary_embedding/ascend/rounds/report_001.md) · [maca final summary](music_flamingo_rotary_embedding/maca/final_summary.md) |
| `mm_encoder_attention` | [base.py](mm_encoder_attention/base.py) | [bi150](mm_encoder_attention/bi150/) · [ascend](mm_encoder_attention/ascend/) · [maca](mm_encoder_attention/maca/) | [bi150 final summary](mm_encoder_attention/bi150/final_summary.md) · [ascend r001 report](mm_encoder_attention/ascend/rounds/report_001.md) · [maca final summary](mm_encoder_attention/maca/final_summary.md) |
| `mhc_post_layer_mix` | [base.py](mhc_post_layer_mix/base.py) | [bi150](mhc_post_layer_mix/bi150/) · [ascend](mhc_post_layer_mix/ascend/) · [maca](mhc_post_layer_mix/maca/) | [bi150 final summary](mhc_post_layer_mix/bi150/final_summary.md) · [ascend r001 report](mhc_post_layer_mix/ascend/rounds/report_001.md) · [maca final summary](mhc_post_layer_mix/maca/final_summary.md) |
| `mhc_head_compute_mix` | [base.py](mhc_head_compute_mix/base.py) | [bi150](mhc_head_compute_mix/bi150/) · [ascend](mhc_head_compute_mix/ascend/) · [maca](mhc_head_compute_mix/maca/) | [bi150 final summary](mhc_head_compute_mix/bi150/final_summary.md) · [ascend r001 report](mhc_head_compute_mix/ascend/rounds/report_001.md) · [maca final summary](mhc_head_compute_mix/maca/final_summary.md) |
| `centre_random_augmentation` | [base.py](centre_random_augmentation/base.py) | [bi150](centre_random_augmentation/bi150/) · [ascend](centre_random_augmentation/ascend/) | [bi150 final summary](centre_random_augmentation/bi150/final_summary.md) · [ascend r001 report](centre_random_augmentation/ascend/rounds/report_001.md) |
| `mhc_head_compute_mix_backward` | [base.py](mhc_head_compute_mix_backward/base.py) | [bi150](mhc_head_compute_mix_backward/bi150/) · [ascend](mhc_head_compute_mix_backward/ascend/) | [bi150 final summary](mhc_head_compute_mix_backward/bi150/final_summary.md) · [ascend r001 report](mhc_head_compute_mix_backward/ascend/rounds/report_001.md) |

## 维护

新增/完结 campaign 时更新本表。可考虑用 `scripts/update_matrix.py` 从各
`project.md` / `outcome.md` 自动汇总（尚未实现）。

# 赛道一：Triton 算子优化

**官方文档**：https://aicarrier.feishu.cn/wiki/YS3OwsOEGiKz2jkaMh9cyGVvngb（需登录）

## 赛题

在多个国产硬件后端上，用 Triton 实现并优化 **10 个算子**，以 `auto_bench.py`
（v0 PyTorch 参考 vs v1 Triton 候选）的 wall time 为度量。

## 任务清单（task 编号 ↔ 算子目录）

10 个任务的 base.py 已全部落盘（`kernels/track1-triton/<算子>/base.py`，
设备无关共享参考）。优化进度见
[矩阵表](../../kernels/track1-triton/README.md)。

| task | 算子目录 | 语义 |
|---|---|---|
| 1 | `groupedtopk` | 分组 top-k 专家路由（已完结：mlu 6.56x、s60 1.68x、maca 3.29x、bi150 1.71x） |
| 2 | `flexattention` | 因果 SDPA 融合（已完结：mlu 7.08x） |
| 3 | `fused_moe` | MoE 路由 + per-expert GEMM（已完结：mlu 50.4x） |
| 4 | `sparse_pooler` | SPLADE 稀疏池化（已完结：mlu 1.60x） |
| 5 | `music_flamingo_rotary_embedding` | 音乐位置编码（batch 时间 + 序列时间，输出 cos/sin） |
| 6 | `mm_encoder_attention` | MMEncoderAttention：多模态编码器注意力（view/transpose + `F.scaled_dot_product_attention`） |
| 7 | `mhc_post_layer_mix` | MHC 后层混合：einsum(`abmn,abmc→abnc`) + post_layer_mix 加权 |
| 8 | `mhc_head_compute_mix` | sigmoid pre/post + comb 矩阵 Sinkhorn 归一化（20 轮迭代） |
| 9 | `centre_random_augmentation` | 蛋白坐标中心化 + 随机刚体增广（四元数旋转矩阵） |
| 10 | `mhc_head_compute_mix_backward` | mhc_head_compute_mix 手动反向（grad_input/scale/base） |

> task 5–10 的 base 由任务模板落盘；task 1–4 为仓库既有算子，编号顺序为推断
> （待飞书官方清单确认）。若官方算子名与目录名不一致，以本表为准做映射。

## 后端矩阵（当前）

见 [kernels/track1-triton/README.md](../../kernels/track1-triton/README.md)。
后端编码 ↔ 芯片映射见 [backend-registry.md](../backend-registry.md)。

## 度量口径

- harness：仓库根 `auto_bench.py`（AST loader 加载 v0/v1 文件，`Model` → `ModelNew` 契约）。
- 正确性：`torch.allclose(atol=1e-2, rtol=1e-2)`（浮点）+ 整型精确相等。
- 性能：`--warmup 50 --repeat 100`，取未取整 median wall time；profiler 另记 device 时间。
- 已完结 campaign 的结论见各 `outcome.md` / `project.md`。

## 提交与评审（待飞书规则确认后补充）

# 赛道一：Triton 算子优化

**官方文档**：https://aicarrier.feishu.cn/wiki/YS3OwsOEGiKz2jkaMh9cyGVvngb（需登录）

## 赛题

在多个国产硬件后端上，用 Triton 实现并优化 **10 个算子**，以 `auto_bench.py`
（v0 PyTorch 参考 vs v1 Triton 候选）的 wall time 为度量。

> ⚠️ 10 算子完整清单在飞书文档内，当前无法访问。仓库已有 4 个算子
> （见下方矩阵），其余算子待补充后录入本文件与
> [矩阵表](../../kernels/track1-triton/README.md)。

## 后端矩阵（当前）

见 [kernels/track1-triton/README.md](../../kernels/track1-triton/README.md)。
后端编码 ↔ 芯片映射见 [backend-registry.md](../backend-registry.md)。

## 度量口径

- harness：仓库根 `auto_bench.py`（AST loader 加载 v0/v1 文件，`Model` → `ModelNew` 契约）。
- 正确性：`torch.allclose(atol=1e-2, rtol=1e-2)`（浮点）+ 整型精确相等。
- 性能：`--warmup 50 --repeat 100`，取未取整 median wall time；profiler 另记 device 时间。
- 已完结 campaign 的结论见各 `outcome.md` / `project.md`。

## 提交与评审（待飞书规则确认后补充）

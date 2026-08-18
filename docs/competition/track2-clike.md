# 赛道二：Clike 算子优化

**官方文档**：https://aicarrier.feishu.cn/wiki/MLfLwP3pGiBO8kkKhO9cY7mVn3g（需登录）

## 现状

- 赛道二规则尚未确认（飞书文档需登录）。
- 现有 C-like 工作：`kernels/track2-clike/fused_moe/mlu/bangc/`
  （原 `mlu/fused_moe/bangc/`，fused_moe 调查中的手写 BANG C 探针，结论见
  [fused_moe 赛道一 outcome](../../kernels/track1-triton/fused_moe/mlu/outcome.md)：
  标量 GEMM 反向优化、矩阵单元路径被 P0 blocked）。

## 目录约定（预留）

赛道二 campaign 采用与赛道一相同的算子优先结构：

```
kernels/track2-clike/<算子>/<后端>/
    base.py  baseline_adapter.py  project.md  state/  rounds/  log/
```

后端首次出现时，确认是否存在对应的 C-like target profile
（`skills/kernel-opt-loop/prompts/coder_targets/`，目前仅 triton 系列）。

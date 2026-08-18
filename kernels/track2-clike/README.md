# 赛道二：Clike（C-like）算子优化

规则确认前的预留区，结构与赛道一一致（算子优先）：

```
kernels/track2-clike/<算子>/<后端>/
```

## 现状

| 算子 | 后端 | 内容 | 来源 |
|---|---|---|---|
| `fused_moe` | `mlu` | [bangc/](fused_moe/mlu/bangc/) 手写 BANG C 探针（标量 GEMM + 矩阵单元调查） | 原 `mlu/fused_moe/bangc/`，fused_moe 赛道一 campaign 产物 |

调查结论（见 [fused_moe 赛道一 outcome](../track1-triton/fused_moe/mlu/outcome.md)）：
手写 BangC 标量 GEMM 是反向优化（比 Triton v5 慢 27x）；矩阵单元路径因 fp32 输出
layout 未公开被 P0 blocked。

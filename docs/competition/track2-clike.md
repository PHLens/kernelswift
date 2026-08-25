# 赛道二：Clike 算子优化

**官方文档**：https://aicarrier.feishu.cn/wiki/MLfLwP3pGiBO8kkKhO9cY7mVn3g（需登录）

## 赛题

C-like（如 BANG C）算子优化。已收到 3 个任务的 PyTorch 参考实现，base 已落盘
（`kernels/track2-clike/<算子>/base.py`，设备无关共享参考）。

| task | 算子目录 | 语义 |
|---|---|---|
| 1 | `sparse_attn` | DeepSeek-V4-Pro sparse attention：top-k 稀疏 KV 注意力，attention sink 仅入 softmax 分母 |
| 2 | `index_topk` | MQA index 模块：压缩 KV 上学习式评分 + top-k 位置选择（含 RoPE、因果 mask） |
| 3 | `sinkhorn_normalize` | Sinkhorn 迭代归一化（softmax + 行/列归一化，doubly stochastic） |

- task1 目标 shape：q `[8,2600,64,128]` bf16，kv `[8,32,128]` bf16，topk=16。
- task2 目标 shape：x `[8,2600,1024]` bf16，qr `[8,2600,256]` bf16，index_topk=128，compress_ratio=4。
- task3 目标 shape：x `[1,1024,4,4]` fp32，repeat=10。

## 目录约定

赛道二采用与赛道一相同的算子优先结构：

```
kernels/track2-clike/<算子>/base.py    # 共享参考（一份，设备无关）
kernels/track2-clike/<算子>/<后端>/    # campaign（baseline_adapter / project.md / state/ / rounds/ / log/）
```

- base.py 均为纯 torch + `device="cuda"` 字符串（harness 自动搬运/重写设备）。
- task2 模板的模块级 `args = ModelArgs(...)` 会被 harness AST 过滤器剥离，已移入
  `get_inputs` / `get_init_inputs` 内部。
- 当前 `kernel-opt-loop` v1 runner 只支持 Python `ModelNew` + Triton 单文件候选；
  C-like 所需的多文件 manifest、原生构建、runner、ABI 和 profiler adapter 尚待实现。
- Skill 扩展方案和实施顺序见 [`../../skills/track2-clike-roadmap.md`](../../skills/track2-clike-roadmap.md)。

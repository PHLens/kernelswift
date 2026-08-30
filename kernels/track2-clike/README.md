# 赛道二：Clike（C-like）算子优化

结构同赛道一（算子优先）：`kernels/track2-clike/<算子>/base.py`（设备无关共享参考）。

## 题目清单（task 编号 ↔ 算子）

| task | 算子目录 | 语义 |
|---|---|---|
| 1 | `sparse_attn` | DeepSeek-V4-Pro sparse attention：top-k 稀疏 KV 注意力 + attention sink（仅入分母） |
| 2 | `index_topk` | MQA index 模块：学习式评分选择 top-k 压缩 KV 位置（压缩 + RoPE + einsum 评分） |
| 3 | `sinkhorn_normalize` | Sinkhorn 迭代归一化（doubly stochastic：softmax + 行列归一化 ×repeat） |

- base.py 均为纯 torch + `device="cuda"` 字符串，harness 自动搬运/重写设备。
- task2 模板的模块级 `args = ModelArgs(...)` 会被 harness AST 过滤器剥离，
  已移入 `get_inputs` / `get_init_inputs` 内部；`.cuda()` 统一改为 `device="cuda"`。
- 优化实现（C-like / Triton）由对应 skill 流程生成，后端目录在建 campaign 时创建。

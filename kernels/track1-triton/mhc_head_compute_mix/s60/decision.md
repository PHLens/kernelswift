# mhc_head_compute_mix — Design Decision

## 1. 算子语义

sigmoid 门控 + 4x4 矩阵行/列归一化的 Sinkhorn 迭代（20 轮）。

### 输入
| 参数 | shape | dtype |
|---|---|---|
| `mixes` | `(b, s, mix_hc)` = `(2, 8, 24)` | float32 |
| `hc_scale` | `(3,)` = `[0.5, 0.25, 1.0]` | float32 |
| `hc_base` | `(mix_hc,)` = `(24,)` | float32 |

其中 `hc = hc_mult = 4`，`mix_hc = (2 + hc) * hc = 24`。

### 输出（返回三元组）
1. `pre`: `(b, s, hc)` = `(2, 8, 4)` float32
2. `post`: `(b, s, hc)` = `(2, 8, 4)` float32
3. `comb`: `(b, s, hc, hc)` = `(2, 8, 4, 4)` float32

### 精确数学
设 `x = mixes.reshape(-1, mix_hc)`（展平 batch*seq 维为 `R = b*s = 16` 行）。`mixes` 分为三段：
- `x[:, :hc]`（前 4 列）→ pre 门控
- `x[:, hc:2*hc]`（第 4-8 列）→ post 门控
- `x[:, 2*hc:]`（后 16 列）→ comb 原始矩阵

```
s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

pre  = sigmoid(x[:, :hc]        * s0 + base[:hc].unsqueeze(0)) + eps          # [R, hc]
post = 2 * sigmoid(x[:, hc:2hc] * s1 + base[hc:2hc].unsqueeze(0))            # [R, hc]

raw  = x[:, 2hc : 2hc + hc*hc]                                               # [R, hc*hc]
comb = raw.view(-1, hc, hc) * s2 + base[2hc:].view(1, hc, hc)                # [R, hc, hc]

# 初始化 + Sinkhorn（20 轮）
row_max = comb.amax(dim=-1, keepdim=True)
comb = exp(comb - row_max)                                                   # 数值稳定
comb = comb / comb.sum(dim=-1, keepdim=True) + eps                           # 行归一
comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)                         # 列归一
for _ in range(sinkhorn_iters - 1):    # 19 次额外迭代
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)                     # 行归一
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)                     # 列归一
```
注意 eps 加法位置：行归一后 `+ eps`（与分母的 `+ eps` 不同，分母的 eps 在求和后加）。仔细看：
- 行归一：`comb = comb / row_sum + eps`（分子不变，整体加 eps）
- 列归一：`comb = comb / (col_sum + eps)`（分母加 eps）

（循环体内两次归一写法与初始化后那两行一致。）

## 2. Triton 设计

关键：Sinkhorn 是对 **4x4 小矩阵**做行/列归一迭代，每个 `(b,s)` 位置独立。矩阵尺寸极小（hc=4），非常适合 **per-row program + 静态展开**。

### Kernel 划分
**单 kernel，grid = (R,)**，`R = b*s = 16` 个 program，每个 program 处理一个 4x4 的 comb 矩阵 + 对应的 pre/post 门控。

每个 program 内：
1. `row = program_id(0)`，`b = row // s`，`s_idx = row % s`（反解用于输出定位）。
2. load `x[row, :]`（24 个元素），`hc_scale`（3 个），`hc_base`（24 个）。
3. **pre/post 门控**：静态展开 4 列，`pre_c = sigmoid(x_c*s0 + base_c) + eps`，`post_c = 2*sigmoid(x_{hc+c}*s1 + base_{hc+c})`。用 `tl.arange(0, hc)` 一次性向量化即可（hc=4 作为 constexpr）。
4. **comb 矩阵**：`comb[i,j] = x[2hc + i*hc + j] * s2 + base[2hc + i*hc + j]`，构造 4x4（用 `tl.arange(0,4)[:,None]` 与 `[None,:]` 二维索引）。
5. **数值稳定初始化**：`row_max = tl.max(comb, axis=1, keepdim=True)`；`comb = exp(comb - row_max)`。
6. **Sinkhorn 20 轮**：用 `tl.static_range` 或 Python `range` 展开 20 次，每轮先 `comb /= (comb.sum(axis=1, keepdim=True) + eps)`（行）再 `comb /= (comb.sum(axis=0, keepdim=True) + eps)`（列）。注意 base 的精确 eps 加法位置：
   - 行归一后整体 `+ eps`（base 初始化阶段是 `comb = comb / row_sum + eps`）
   - 列归一分母 `+ eps`
   为严格对齐，实现需复刻这个"行归一整体加 eps、列归一分母加 eps"的细节。

   仔细核对 base 代码：
   ```
   comb = comb / comb.sum(dim=-1, keepdim=True) + eps          # 行：整体 +eps
   comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)        # 列：分母 +eps
   for ...:
       comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)    # 行：分母 +eps  ← 注意与初始化不同！
       comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)    # 列：分母 +eps
   ```
   **关键差异**：初始化阶段的行归一是 `整体 + eps`，而循环内行归一是 `分母 + eps`。实现时必须精确复刻这个不对称，否则数值不匹配。
7. store `pre[row]`、`post[row]`、`comb[row]`。

### Host 辅助
- `mixes.reshape(-1, mix_hc)` 可在 host 用 `.reshape` 完成（简单 view），或直接在 kernel 内按展平索引 load。建议 host 先 reshape，简化 kernel 索引。
- 无其他 host 辅助；20 轮迭代在 kernel 内完成，是真正核心计算。

## 3. 关键难点
- **Sinkhorn 迭代的 eps 不对称**：初始化行归一 `整体+eps` vs 循环内 `分母+eps`。这是最大正确性陷阱，必须逐字符对齐 base。
- **4x4 小矩阵**：hc=4 作为 constexpr，用 `tl.arange(0,4)` 二维索引构造矩阵，`tl.sum(axis=...)` 做行/列归约，`tl.max` 做行最大。无需 `tl.dot`。
- **迭代展开**：20 轮用 Python `for` 循环（在 jit 内用 `tl.static_range(0, sinkhorn_iters)` 或直接 `range`，因为 sinkhorn_iters 是 constexpr）。
- **三段切分**：`mixes` 的 24 列分为 pre(4)/post(4)/comb(16) 三段，索引边界 `:hc`、`hc:2hc`、`2hc:` 要精确。

## 4. 正确性风险点
- **eps 不对称**（上述）：最大的精度风险，务必复刻。
- **dtype**：全程 float32，无精度损失，atol 建议 1e-5。
- **sigmoid 数值**：`tl.exp(-z)` 计算 sigmoid，与 torch 一致。
- **comb 数值稳定**：`exp(comb - row_max)` 后值域 (0,1]，归一化后无溢出风险。
- **边界**：R=16 个 program，每个独立，无跨 program 依赖，天然正确。
- **输出 shape**：`pre/post` 为 `(2,8,4)`，`comb` 为 `(2,8,4,4)`，需 `.view(b, s, hc)` 与 `.view(b, s, hc, hc)` 还原。

## 5. ModelNew 接口设计
- `ModelNew.__init__(hc_mult=4, sinkhorn_iters=20, eps=1e-6)`，与 base `Model.__init__` 签名一致，存为实例属性（非 buffer）。
- `forward(mixes, hc_scale, hc_base)` 返回 `(pre, post, comb)` 三元组。
- `get_inputs()`：复用 base（含 `torch.manual_seed(0)`），`mixes` `(2,8,24)` float32，`hc_scale` `[0.5,0.25,1.0]`，`hc_base` `(24,)` float32。
- `get_init_inputs()`：返回 `[4, 20, 1e-6]`。
- **无 state_dict 匹配问题**：base `Model` 无 buffer/parameter（只有 Python 属性）。

# flexattention — Design Decision

## 1. 算子语义

**Causal** 缩放点积注意力（flexattention，is_causal=True）。

### 输入
| 参数 | shape | dtype |
|---|---|---|
| `query` | `(num_tokens, num_heads, head_size)` = `(83, 8, 64)` | float16 |
| `key` | `(num_tokens, num_kv_heads, head_size)` = `(83, 8, 64)` | float16 |
| `value` | `(num_tokens, num_kv_heads, head_size)` = `(83, 8, 64)` | float16 |

其中 `num_heads=8, head_size=64, num_kv_heads=8`（MHA），`num_tokens=83`。

### 输出
- `out`: `(num_tokens, num_heads*head_size)` = `(83, 512)` float16

### 精确数学
注意：输入布局与 mm_encoder_attention 不同，这里是 `[num_tokens, num_heads, head_size]`（无 batch 维）。

```
q = query.unsqueeze(0).transpose(1,2)   # [1, H, num_tokens, d]
k = key.unsqueeze(0).transpose(1,2)     # [1, H, num_tokens, d]
v = value.unsqueeze(0).transpose(1,2)   # [1, H, num_tokens, d]
# num_kv_heads == num_heads，无需 repeat_interleave
out = scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
    # [1, H, num_tokens, d]
out = out.squeeze(0).transpose(0,1).reshape(num_tokens, H*d)
```

**Causal mask**：位置 i 只能 attend 到位置 j <= i（下三角 mask）。`scores[i, j] = -inf` 当 `j > i`。

## 2. Triton 设计

**正确性优先的手写 causal softmax attention**，与 mm_encoder_attention 结构一致，但增加 causal 下三角 mask。无 `tl.dot`，GEMM 用 `tl.sum` 展开。

### Kernel 划分
**grid = (num_heads,)**（无 batch 维，batch=1 隐式），每个 program 处理一个 head。

每 program（给定 h）：
1. 定位该 head 的 q/k/v：`q = query[:, h, :]` 形状 `[num_tokens, d]`，`k = key[:, h, :]`、`v = value[:, h, :]` 形状 `[num_tokens, d]`。
2. `BLOCK = 128`（覆盖 num_tokens=83），`offs_m = arange(0,128)`，`offs_n = arange(0,128)`，`offs_d = arange(0,64)`。
3. load q/k/v：`q[offs_m, offs_d]`（mask_m = offs_m < 83），`k[offs_n, offs_d]`（mask_n = offs_n < 83），`v[offs_n, offs_d]`。
4. **scores**：`scores[m, n] = scale * sum_d(q[m,d] * k[n,d])`，用 `tl.sum(q[:,None,:] * k[None,:,:], axis=2)`，fp32。
5. **causal mask**：`causal_mask = offs_n[None,:] <= offs_m[:,None]`（下三角），`scores = where(causal_mask & mask_n, scores, -inf)`。
   - 注意还要 mask 越界的 n（`mask_n`），以及越界的 m（`mask_m`，用于最终 out mask）。
6. **softmax**：`m = max(scores, axis=1)`；`p = exp(scores - m)`；`l = sum(p, axis=1)`；`attn = p / l`。
   - 注意：对于越界的 query 行（offs_m >= 83），该行所有 scores 都是 -inf，`max` 会是 -inf，`exp(-inf - (-inf))` = `exp(nan)` → 需小心。处理方式：对 mask_m 外的行，直接令输出为 0（`tl.where(mask_m, out, 0)`），且用 `tl.where` 保护 `-inf` 行：可将 `scores - m` 中 -inf 项处理为 0。建议用 `scores = where(valid, scores, -1e30)` 且对全 -inf 行，`m = max` 后 `p = exp(scores - m)`，-inf - (-inf) = nan。稳妥做法：`m = where(mask_m, m, 0)`，`p = where(valid, exp(scores - m), 0)`，`l = where(mask_m, sum(p), 1)`。
7. **out**：`out[m, :] = sum_n(attn[m,n] * v[n,:])`，`tl.sum(attn[:,:,None] * v[None,:,:], axis=1)`，fp32。
8. store：`out.to(fp16)`，`where(mask_m, out, 0)`。

### Host 辅助
- `unsqueeze(0).transpose(1,2)` 重排：本算子输入已是 `[num_tokens, H, d]`，无需转置（直接按 `[num_tokens, H, d]` 布局索引），只需在 host 做最终 `.reshape(num_tokens, H*d)`。
- `num_kv_heads == num_heads`，无需 repeat_interleave（base 代码该分支不触发）。

## 3. 关键难点
- **causal mask 实现**：用 `offs_n <= offs_m` 构造下三角布尔 mask，与越界 mask 取 `&`，越界/上三角位置置 `-inf`。
- **全 -inf 行的 softmax 数值安全**：越界 query 行会触发 `-inf - (-inf) = nan`，必须用 `tl.where` 保护（mask_m 外的行直接输出 0，或对 max/exp/sum 逐项保护）。
- **`tl.dot` 不可用**：QK^T 与 PV 用 `tl.sum` 展开，d=64、seq=83 计算量小。
- **无 batch 维**：相比 mm_encoder_attention，少了 bsz 维，grid 更简单（只按 head 划分）。

## 4. 正确性风险点
- **causal 方向**：必须 `j <= i`（下三角），不能写反成 `j >= i`。
- **dtype 精度**：fp16 输入 → fp32 计算 → fp16 输出，atol 建议 1e-2。
- **全 -inf 行 nan**（上述）：这是 causal + mask 组合下最易出错点，务必测试 `num_tokens=83` 非 128 整倍数的情况（越界行）。
- **scale**：`1/sqrt(64) = 0.125`。
- **softmax 与 torch 对齐**：torch 的 SDPA causal 用 `exp(x - max)`，手写一致即可。
- **输出布局**：`[num_tokens, H*d]`，reshape 前是 `[num_tokens, H, d]`（squeeze + transpose(0,1) 后），最终 `reshape(num_tokens, H*d)` 需保证内存布局连续（可 `.contiguous().reshape`）。

## 5. ModelNew 接口设计
- `ModelNew.__init__(num_heads=8, head_size=64, scale=None, num_kv_heads=8)`，`scale = scale or 1/sqrt(head_size)`，与 base 一致。
- `forward(query, key, value)` 返回 `(num_tokens, num_heads*head_size)`。
- `get_inputs()`：复用 base，fp16，`(83, 8, 64)`。
- `get_init_inputs()`：返回 `[8, 64, None, 8]`。
- **无 state_dict 匹配问题**：base 无 buffer/parameter。

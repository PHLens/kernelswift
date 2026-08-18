# mm_encoder_attention — Design Decision

## 1. 算子语义

多头缩放点积注意力（**非 causal**），SDPA 标准形式。

### 输入
| 参数 | shape | dtype |
|---|---|---|
| `query` | `(bsz, q_len, num_heads*head_size)` = `(2, 83, 512)` | float16 |
| `key` | `(bsz, kv_len, num_kv_heads*head_size)` = `(2, 83, 512)` | float16 |
| `value` | `(bsz, kv_len, num_kv_heads*head_size)` = `(2, 83, 512)` | float16 |

其中 `num_heads=8, head_size=64, num_kv_heads=8`（MHA，非 GQA），`bsz=2, q_len=kv_len=83`。

### 输出
- `out`: `(bsz, q_len, num_heads*head_size)` = `(2, 83, 512)` float16

### 精确数学
```
q = query.view(bsz, q_len, num_heads, head_size).transpose(1,2)  # [bsz, H, q_len, d]
k = key.view(bsz, kv_len, num_kv_heads, head_size).transpose(1,2)  # [bsz, H, kv_len, d]
v = value.view(bsz, kv_len, num_kv_heads, head_size).transpose(1,2) # [bsz, H, kv_len, d]

scale = 1/sqrt(head_size) = 1/8
scores = q @ k^T * scale                    # [bsz, H, q_len, kv_len]   (无 mask，非 causal)
attn   = softmax(scores, dim=-1)            # 沿 kv_len 维
out    = attn @ v                           # [bsz, H, q_len, d]
out    = out.transpose(1,2).reshape(bsz, q_len, H*d)
```
**非 causal**：无任何 mask，每个 query 位置 attend 到所有 kv 位置（包括未来）。

## 2. Triton 设计

**正确性优先的手写 softmax attention**：不使用 `tl.dot`（GCU Unknown），GEMM 用 `tl.sum` 展开，慢但正确。标准 FlashAttention 分块结构，但简化（q_len=kv_len=83 很小，可整行加载）。

### Kernel 划分
**grid = (bsz * num_heads,)**，每个 program 处理一个 `(batch, head)` 的注意力。

每 program（给定 b, h）：
1. 定位 q/k/v 该 head 的切片：`q_bh = q[b, h, :, :]` 形状 `[q_len, d]`，`k_bh = k[b, h, :, :]`、`v_bh = v[b, h, :, :]` 形状 `[kv_len, d]`。
2. **分块（tile）遍历 query**：`BLOCK_M` 个 query 行一组，`BLOCK_N` 个 kv 行一组，用 FlashAttention 的 online-softmax 累加器。
   - 由于 seq_len=83 较小，也可直接：对每个 query 行（`BLOCK_M=1` 或整块），遍历 kv 行计算 scores，softmax，再乘 v 累加。
3. **scores 计算**（无 `tl.dot`）：`scores[m, n] = scale * sum_d(q[m, d] * k[n, d])`，用 `tl.sum(q_block[:, None, :] * k_block[None, :, :], axis=2)` 展开点积。
4. **softmax**（非 causal，无 mask）：`m = max(scores, axis=-1)`，`exp(scores - m)`，`l = sum(exp)`，`attn = exp / l`。
5. **out**：`out[m, :] = sum_n(attn[m, n] * v[n, :])`，用 `tl.sum(attn[:, :, None] * v_block[None, :, :], axis=1)`。
6. 由于 seq 只有 83，**最简单正确方案**：BLOCK_M 覆盖整个 q_len（83→128 补齐 mask），BLOCK_N 覆盖整个 kv_len（83→128 补齐），一次算出完整 `scores[128,128]` 矩阵（mask 掉越界位置），softmax 后乘 v。无需 online-softmax 分块，直接全量 softmax，最不易出错。

### 推荐实现（正确性优先，seq 小直接全量）
- `grid = (bsz * num_heads,)`，`BLOCK = 128`（覆盖 83）。
- 每 program：
  ```
  offs_m = arange(0, 128); mask_m = offs_m < q_len
  offs_n = arange(0, 128); mask_n = offs_n < kv_len
  offs_d = arange(0, 64)   # head_size=64
  q = load(q_ptr + b*H*q_len*d + h*q_len*d + offs_m[:,None]*d + offs_d[None,:], mask=mask_m[:,None])
  k = load(k_ptr + ... offs_n[:,None]*d + offs_d[None,:], mask=mask_n[:,None])
  v = load(v_ptr + ..., mask=mask_n[:,None])
  scores = scale * tl.sum(q[:,None,:] * k[None,:,:], axis=2)   # [128,128] fp32
  scores = tl.where(mask_n[None,:], scores, -inf)              # 非 causal，只 mask 越界 kv
  m = tl.max(scores, axis=1, keepdim=True)
  p = tl.exp(scores - m)
  l = tl.sum(p, axis=1, keepdim=True)
  attn = p / l                                                # [128,128]
  out = tl.sum(attn[:,:,None] * v[None,:,:], axis=1)          # [128,64]
  out = tl.where(mask_m[:,None], out, 0.0)
  store(out_ptr + ... , out.to(fp16))
  ```
  - `d=64` 无需 mask（head_size 固定 64，`tl.arange(0,64)` 恰好）。

### Host 辅助
- `view/transpose` 重排：可 host 预先 `.transpose(1,2)` 得到 `[bsz, H, seq, d]` 连续布局，简化 kernel 索引；或在 kernel 内用 strided load。建议 host 先 `q = query.view(bsz, q_len, H, d).transpose(1,2).contiguous()` 等。
- 输出 reshape：kernel 写 `[bsz, H, q_len, d]`，host 再 `.transpose(1,2).reshape(bsz, q_len, H*d)`。

## 3. 关键难点
- **`tl.dot` 不可用**：两个 GEMM（QK^T 与 PV）都用 `tl.sum` 展开。d=64 时 QK^T 需 64 次乘加（用 `tl.sum` 沿 axis=2），PV 需对 kv_len=128 求和。计算量可接受（正确性优先）。
- **数值稳定性**：softmax 减去行最大值。用 fp32 累加（q/k/v 是 fp16，load 后 `.to(tl.float32)`）。
- **非 causal**：无需 mask 上三角，只 mask 越界的 kv（`offs_n < kv_len`）。与 flexattention 的 causal 形成对比。

## 4. 正确性风险点
- **dtype 精度**：输入 fp16，计算 fp32，输出 fp16。base 的 `scaled_dot_product_attention` 在 fp16 下内部可能用 fp32 累加，误差 ~1e-3 量级，atol 建议 1e-2（相对 1e-2）稳过。若评测用 fp16 输出精确对比，需注意 softmax 实现与 torch 的微小差异，建议宽松容差。
- **scale**：`1/sqrt(64) = 0.125`，显式乘。
- **越界 mask**：BLOCK=128 > 83，q/k/v 的 load 都要 mask `offs < 83`，越界填 0；scores 越界 kv 置 `-inf` 使其 softmax 后为 0；out 越界 query 置 0。
- **softmax 实现差异**：torch 的 softmax 是 `exp(x - max) / sum(exp(x - max))`，与手写一致。
- **无 GQA**：num_kv_heads == num_heads == 8，无需 repeat_interleave 逻辑。

## 5. ModelNew 接口设计
- `ModelNew.__init__(num_heads=8, head_size=64, num_kv_heads=8)`，与 base 一致，`self.scale = 1/sqrt(head_size)`。
- `forward(query, key, value)` 返回 `(bsz, q_len, num_heads*head_size)`。
- `get_inputs()`：复用 base，fp16，`(2,83,512)`。
- `get_init_inputs()`：返回 `[8, 64, 8]`。
- **无 state_dict 匹配问题**：base 无 buffer/parameter（num_heads/head_size/num_kv_heads/scale 均为属性/常量）。

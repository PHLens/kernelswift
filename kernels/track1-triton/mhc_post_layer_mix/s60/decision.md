# mhc_post_layer_mix — Design Decision

## 1. 算子语义

Einsum 收缩 + elementwise 融合，输出 bf16。

### 输入
| 参数 | shape | dtype |
|---|---|---|
| `x` | `(n0, n1, h)` = `(2, 4096, 1280)` | bfloat16 |
| `residual` | `(n0, n1, mhc_mult, h)` = `(2, 4096, 4, 1280)` | bfloat16 |
| `post_layer_mix` | `(n0, n1, mhc_mult, 1)` = `(2, 4096, 4, 1)` | float32 |
| `comb_res_mix` | `(n0, n1, mhc_mult, mhc_mult)` = `(2, 4096, 4, 4)` | float32 |

### 输出
- `out`: `(n0, n1, mhc_mult, h)` bfloat16

### 精确数学
```
term2 = einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
      # comb_res_mix[a,b,m,n] 与 residual[a,b,m,c] 对 m 求和 -> [a,b,n,c]
      # 即 term2[b, n, c] = sum_m comb_res_mix[b, m, n] * residual[b, m, c]
out   = (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
      # x.float() -> [a,b,h]，unsqueeze(-2) -> [a,b,1,h]
      # 乘 post_layer_mix[a,b,mhc_mult,1] 广播到 [a,b,mhc_mult,h]
      # 加 term2，再 cast 到 bf16
```
注意：`mhc_mult = 4`，`h = 1280`。einsum 中 `mhc_mult` 是收缩维（尺寸 4），`h` 是保留维（尺寸 1280）。

**广播规则**：`x.unsqueeze(-2)` 形状 `[a,b,1,h]` 与 `post_layer_mix` `[a,b,mhc_mult,1]` 相乘，广播得到 `[a,b,mhc_mult,h]`。

## 2. Triton 设计

核心是 **einmm 收缩（小维 mhc_mult=4）+ elementwise 融合**。GEMM 收缩维只有 4，`tl.dot` 在 GCU 上不可用，用 `tl.sum` 展开即可（收缩维 4 极小，展开代价可忽略）。

### Kernel 划分
采用 **per-(b, c) 平铺**：对每个 batch `b`（n0*n1 = 8192）与每个输出通道 `c`（h = 1280）分块。

- 外层 grid 平铺 `(n0*n1*mhc_mult)` 个 program，每个 program 处理一个 `(b, n)` 位置（n 是 mhc_mult 输出索引），并处理 h 维的一个 BLOCK。
- 更简单方案：`grid = (n0 * n1 * mhc_mult,)`，每个 program 对应一个 `(b, n)`，用 `BLOCK_H` 平铺 h。
  - `b = pid // mhc_mult`，`n = pid % mhc_mult`（这里的 n 是 mhc_mult 输出通道）。
  - `h_idx = arange(0, BLOCK_H)`。

### 每 program 计算（term2 的 einsum）
对每个 `(b, n, c)`：
```
acc = 0  (float32)
for m in static_range(0, mhc_mult):   # 收缩维，4 次
    cmm = load(comb_res_mix[b, m, n])              # 标量
    res = load(residual[b, m, c_block])            # [BLOCK_H] bf16 -> fp32
    acc += cmm * res
# term2[b, n, c_block] = acc
x_val   = load(x[b, c_block])                      # [BLOCK_H] bf16 -> fp32
plm     = load(post_layer_mix[b, n, 0])            # 标量
out_val = x_val * plm + acc                        # [BLOCK_H] fp32
store(out[b, n, c_block], out_val.to(bf16))
```
其中 `x_val * plm` 对应 `x.unsqueeze(-2) * post_layer_mix`（注意：`post_layer_mix` 形状 `[...,mhc_mult,1]`，最后一维为 1，所以 `plm = post_layer_mix[b, n, 0]`）。

### Host 辅助
- 无复杂 host 辅助；输入 bf16 在 kernel 内 `.to(tl.float32)` 后计算，输出 `.to(tl.bfloat16)`。
- `comb_res_mix` 是 float32，直接 load。

## 3. 关键难点
- **收缩维极小（4）**：用 `tl.static_range` 展开 4 次 `tl.sum` 式乘加，避免 `tl.dot`（GCU Unknown）。
- **bf16 精度**：base 中 `residual.float()` 后参与 einsum，`x.float()` 后参与 elementwise，最终 `.bfloat16()` 截断。Triton 实现必须**全程 float32 累加**，只在最终 store 时 cast 到 bf16，与 base 语义一致。
- **索引反解**：`b = pid // mhc_mult`，`n = pid % mhc_mult`，注意区分 mhc_mult（收缩/输出通道维）与 h（feature 维）。

## 4. 正确性风险点
- **dtype 精度**：bf16 输入，float32 计算，bf16 输出。与 base 完全一致的 dtype 流转，误差主要来自 bf16 量化（相对误差 ~2^-8），atol 建议 1e-2（相对 1e-2）可稳过。
- **收缩顺序**：einsum 是对 `m` 求和，务必用 `m` 作为收缩索引，`n` 作为输出通道，避免混淆。
- **广播**：`post_layer_mix[..., mhc_mult, 1]` 最后一维是 1，取 `[b, n, 0]` 标量，与 `x_val` 逐元素相乘。
- **边界**：h=1280，若 BLOCK_H 不能整除，需 mask。建议 BLOCK_H=128，1280/128=10 整除，grid 增加 h 平铺维度，或单个 program 内循环。为简单，可用 `BLOCK_H = 128` 且 `grid = (n0*n1*mhc_mult, h // BLOCK_H)` 两维 grid。
- **输出 shape**：必须 `(2, 4096, 4, 1280)`。

## 5. ModelNew 接口设计
- `ModelNew.__init__()` 无参数，无 buffer/参数。
- `forward(x, residual, post_layer_mix, comb_res_mix)` 返回单个 tensor（bf16）。
- `get_inputs()`：base 的 `generate_mhc_post_test_data` 返回 5 个，但 `get_inputs` 只取前 4 个 `[x, residual, post_layer_mix, comb_res_mix]`。注意 `get_inputs` 中 `o_grad` 未被使用，ModelNew 的 `get_inputs` 只返回 4 个输入即可。
- `get_init_inputs()`：返回 `[]`。
- **无 state_dict 匹配问题**：base 无 buffer/parameter。

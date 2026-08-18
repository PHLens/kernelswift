# mhc_head_compute_mix_backward — Design Decision

## 1. 算子语义

手工实现 `mhc_head_compute_mix` 的反向传播（sigmoid 门控的梯度）。

### 输入
| 参数 | shape | dtype |
|---|---|---|
| `input_mix` | `(n0, n1, mhc_mult)` = `(2, 1024, 4)` | float32 |
| `mhc_scale` | `(1,)` | float32 |
| `mhc_base` | `(mhc_mult,)` = `(4,)` | float32 |
| `grad_out` | `(n0, n1, mhc_mult)` | float32 |

### 输出（返回三元组）
1. `grad_input_mix`: `(n0, n1, mhc_mult)` float32
2. `grad_mhc_scale`: `(1,)` float32（标量梯度）
3. `grad_mhc_base`: `(mhc_mult,)` float32

### 精确数学
前向中间量：
```
z       = input_mix * mhc_scale + mhc_base      # broadcast: scale(1,) 与 base(mhc_mult,) 沿最后一维广播
sigmoid = sigmoid(z)
```
sigmoid 反向（链式法则，σ'(z) = σ(z)(1-σ(z))）：
```
grad_z          = grad_out * sigmoid * (1 - sigmoid)
grad_input_mix  = grad_z * mhc_scale                    # broadcast scale 沿最后一维
grad_mhc_base   = grad_z.sum(dim=(0,1))                 # -> (mhc_mult,)
grad_mhc_scale  = (grad_z * input_mix).sum(dim=(0,1,2)) # -> 标量 (1,)
```

**广播规则**：`mhc_scale` 形状 `(1,)`，`mhc_base` 形状 `(mhc_mult,)`。在 `input_mix * mhc_scale + mhc_base` 中，`mhc_scale` 广播到所有元素（标量乘法），`mhc_base` 沿最后一维 `(mhc_mult,)` 广播到 `(n0, n1)` 的每个位置。

**Reduction 维度**：
- `grad_mhc_base`：对 batch 两维 `(0,1)` 求和，得到每个 `mhc_mult` 通道的梯度。
- `grad_mhc_scale`：对所有 `(0,1,2)` 求和，得到单个标量。

## 2. Triton 设计

本质是 **elementwise 融合 + 归约**。核心计算是逐元素 `z → sigmoid → grad_z → grad_input_mix`，加上两个归约（对 batch 求和 / 全量求和）。

### Kernel 划分

**Kernel A（elementwise，主 kernel）**：grid 平铺展平后的 `n0*n1*mhc_mult = 8192` 个元素。
- `BLOCK = 256`，`grid = (8192 // BLOCK,)`。
- 每个 program 内：
  1. 用 `offs = pid*BLOCK + arange(BLOCK)`，反解出 `b = offs // (n1*mhc_mult)`、`rem = offs % (n1*mhc_mult)`、`t = rem // mhc_mult`、`c = rem % mhc_mult`（索引反解 b/t/d，参考 rotary_002 风格）。
  2. `load input_mix`、`grad_out`、`mhc_base[c]`、`mhc_scale[0]`。
  3. `z = x * scale + base_c`；`sig = sigmoid(z)`；`gz = grad_out * sig * (1-sig)`。
  4. `store grad_input_mix[offs] = gz * scale`。
  5. 同时把 `gz` 与 `gz * input_mix` 累积进**局部归约**。

**归约处理（正确性优先，允许 host 辅助）**：由于 `grad_mhc_base` 与 `grad_mhc_scale` 都是小输出（`(4,)` 与 `(1,)`），且涉及跨 program 的跨 batch 归约，最稳妥的做法是：
- Triton kernel 只负责计算并写出 `grad_input_mix`（以及可选的逐元素 `gz` 中间量到临时 buffer）。
- `grad_mhc_base = gz.sum(dim=(0,1))` 与 `grad_mhc_scale = (gz * input_mix).sum()` 在 **host 端用 torch** 完成（`torch.sum`），因为这两个输出量级极小（4 个 / 1 个标量），host 归约开销可忽略，且绝对正确、无精度风险。

> 说明：若追求纯 Triton，可用 `tl.atomic_add` 做跨 program 归约，但 GCU 上 atomic 行为与性能风险较大。按"正确性是硬门槛"原则，**主 kernel 用 Triton 完成核心逐元素计算（sig/gz/gi），小归约用 host torch.sum**，forward 中确实有真实 Triton kernel 执行核心计算。

### Host 辅助
- `sigmoid(z)` 等价于 `1/(1+exp(-z))`，在 kernel 内用 `tl.exp` 计算，保证与 torch.sigmoid 数值一致。
- 小归约 `grad_mhc_base` / `grad_mhc_scale` 用 `torch.sum`（见上）。

## 3. 关键难点
- 无显著难点。主要是正确反解展平索引、以及 `mhc_base` 沿最后一维的 gather（`base[c]` 用 `tl.load(mhc_base_ptr + c)`）。
- 保持 float32 全程，无需 dtype 转换。

## 4. 正确性风险点
- **dtype/精度**：全程 float32，与 base 一致，无精度损失。`tl.exp` 与 `torch.sigmoid` 在 float32 下误差极小（~1e-7 量级），atol 建议 1e-5 即可稳过。
- **广播方向**：`mhc_base` 必须按 `c = offs % mhc_mult` 取最后一个通道，不能误取 batch 维度。`mhc_scale` 是标量广播，无歧义。
- **边界**：`n0*n1*mhc_mult = 8192` 能被 `BLOCK=256` 整除，无 mask 需求；但为稳健仍加 `mask = offs < total`。
- **输出 shape**：`grad_mhc_base` 需 `.view(-1)` 为 `(4,)`，`grad_mhc_scale` 需 `.view(1)` 为 `(1,)`，与 base 完全一致。

## 5. ModelNew 接口设计
- `ModelNew.__init__()` 无参数（与 base `Model` 一致），无 register_buffer / 参数。
- `forward(input_mix, mhc_scale, mhc_base, grad_out)` 返回 `(grad_input_mix, grad_mhc_scale, grad_mhc_base)` 三元组，顺序与 base 一致。
- `get_inputs()`：复用 base 的 `[input_mix, mhc_scale, mhc_base, grad_out]`（float32, cuda）。
- `get_init_inputs()`：返回 `[]`。
- **无 state_dict 匹配问题**：base `Model` 无任何 buffer/parameter，`ModelNew` 同样为空。

# centre_random_augmentation — Design Decision

## 1. 算子语义

扩散采样中的随机刚体变换：中心化 → 随机旋转（四元数）+ 随机平移 → 可选 mask。

### 输入
| 参数 | shape | dtype |
|---|---|---|
| `x_input_coords` | `(N_atom, 3)` = `(256, 3)` | float32 |
| `mask` | `(N_atom,)` 0/1 | float32 |

### 输出
- `x_aug`: `(n_sample, N_atom, 3)` = `(4, 256, 3)` float32

### 精确数学
```
if mask is None:
    center = x_input_coords.mean(dim=-2, keepdim=True)      # [1,3]
else:
    m = mask.unsqueeze(-1)
    center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + eps)

x = x_input_coords - center                                  # [N_atom, 3] 中心化
x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()     # [n_sample, N_atom, 3]

R = random_rotation_matrices(n_sample)                       # [n_sample, 3, 3] 四元数随机旋转
T = s_trans * torch.randn(n_sample, 3)                       # [n_sample, 3] 随机平移
x = rot_vec_mul(R[:,None,:,:].expand(-1, N_atom, -1, -1), x) + T[:,None,:]
    # rot_vec_mul: 每个 (sample, atom) 做 R[3,3] @ x[3] 矩阵乘
if mask is not None:
    x = x * mask[None,:,None]                                # mask 广播
```

**四元数 → 旋转矩阵**（`random_rotation_matrices`）：
```
u1, u2, u3 = rand(n)  # 均匀 [0,1)
q1 = sqrt(1-u1)*sin(2π*u2);  q2 = sqrt(1-u1)*cos(2π*u2)
q3 = sqrt(u1)*sin(2π*u3);    q4 = sqrt(u1)*cos(2π*u3)
x,y,z,w = q1,q2,q3,q4        # 四元数 (x,y,z,w)

xx, yy, zz = x*x, y*y, z*z
xy, xz, yz = x*y, x*z, y*z
wx, wy, wz = w*x, w*y, w*z

R = [ 1-2(yy+zz), 2(xy-wz), 2(xz+wy)
      2(xy+wz),   1-2(xx+zz), 2(yz-wx)
      2(xz-wy),   2(yz+wx),  1-2(xx+yy) ].reshape(n,3,3)
```

## 2. Triton 设计

**随机数在 host 生成**（GCU 上无法在 kernel 内生成 torch.rand），Triton kernel 只负责确定性计算：四元数→旋转矩阵、rot_vec_mul（矩阵乘向量）、以及中心化/平移/mask。

### Kernel 划分

**Kernel 1：四元数 → 旋转矩阵**（`grid = (n_sample,)`，每 program 一个旋转矩阵）
- 输入：`u1, u2, u3`（host 生成的 `[n_sample]` 随机数）——或者直接在 host 用 torch 算好四元数 q1..q4 再传入 kernel，进一步简化。
- 每 program：load u1/u2/u3，算 q1..q4，再算 9 个矩阵元素，store 到 `R[n_sample, 3, 3]`。
- 这里也可以用 host torch 直接算旋转矩阵（`random_rotation_matrices` 本质是 elementwise），但为体现 Triton 核心计算，建议用 Triton kernel 从四元数算旋转矩阵。

**Kernel 2：中心化 + rot_vec_mul + 平移 + mask**（`grid = (n_sample, ceil(N_atom/BLOCK))` 或 `grid=(n_sample*N_atom,)`）
- 每 program 处理一个 `(sample, atom_block)`：
  1. load 中心化后的 `x_centered[atom]`（host 预先算好 `x_centered = x_input_coords - center`，或 kernel 内算）。
  2. load `R[sample, :, :]`（3x3）。
  3. `rot = R @ x`（3 维矩阵乘向量，用 3 次点积静态展开）。
  4. `+ T[sample, :]`（平移）。
  5. 若 mask：`* mask[atom]`。
  6. store 到 `x_aug[sample, atom, :]`。

### Host 辅助（正确性优先）
- **随机数**：`u1,u2,u3 = torch.rand(n_sample)`、`T = s_trans * torch.randn(n_sample, 3)` 在 host 生成，传入 kernel。
- **中心化**：`center = (x_input_coords * m).sum(-2) / (m.sum(-2) + eps)` 在 host 用 torch 计算（小量，绝对正确），得到 `x_centered = x_input_coords - center`，传入 kernel。
- 四元数→旋转矩阵：可由 host 直接用 torch elementwise 完成（等价于 base 的 `random_rotation_matrices`），但按"forward 必须有 Triton 核心计算"，建议将 **rot_vec_mul（旋转+平移+mask）** 作为 Triton kernel 的核心计算，四元数→R 的 elementwise 转换可在 host 或单独 Triton kernel 完成。

> 最终建议：**Triton kernel 承担 rot_vec_mul（含中心化坐标输入、旋转、平移、mask 融合）**，这是核心的刚体变换计算。随机数生成、中心化标量、四元数→矩阵的 elementwise 转换放 host（都无 Triton 必要，且能规避随机数约束）。

## 3. 关键难点
- **随机数**：GCU kernel 内无法 `torch.rand`，必须在 host 生成 u1/u2/u3/T 传入。这是本算子的最大约束。`torch.manual_seed(42)` 在 host 设置，保证可复现。
- **四元数旋转**：9 个矩阵元素由四元数分量组合，纯 elementwise，直接静态展开 9 个表达式。
- **rot_vec_mul**：3x3 @ 3 向量，静态展开 3 个点积（无 tl.dot 需求）。
- **broadcast/expand**：`x` expand 到 `n_sample` 份，kernel 内用 `sample` 索引 + 共享中心化坐标即可，无需真正 expand 内存。

## 4. 正确性风险点
- **随机数一致性**：base 用 `torch.manual_seed(42)`，若 benchmark 对比时 base 与 ModelNew 分别调用随机数，需保证**随机数生成顺序与 base 完全一致**，否则 R/T 不同导致输出不同。**这是最大风险**：正确性对比通常要求相同输入下输出一致，但含随机数的算子在每次 forward 会重新 rand。需与评测方确认：要么固定 seed 且 base 与 ModelNew 用同一 seed、同一随机数序列，要么评测用容差而非精确匹配。
  - 稳妥方案：ModelNew 内部同样 `torch.manual_seed(42)`（在 `get_inputs` 或 forward 内设定），且随机数生成数量/顺序与 base 完全一致（先 3 组 u，再 T）。
- **dtype**：全程 float32，无精度损失。
- **eps 除零**：`center` 分母 `m.sum + eps`，mask 全 0 时不除零。
- **mask 语义**：`mask=None` 时用无偏均值中心化且最后不乘 mask；`mask` 给定（base 的 `get_inputs` 传了全 1 mask）时用加权中心化并最后乘 mask。实现需分支处理（host 判断 mask 是否为 None）。
- **边界**：N_atom=256，n_sample=4，规模小，grid 简单。

## 5. ModelNew 接口设计
- `ModelNew.__init__(n_sample=1, s_trans=1.0, centre_only=False)`，与 base `Model.__init__` 一致。
- `forward(x_input_coords, mask=None)` 返回 `(n_sample, N_atom, 3)`。
  - base 的 `get_inputs` 返回 `[x_input_coords, mask]`（mask 全 1），故 forward 会收到 mask。
  - `centre_only` 分支：若 True，直接返回中心化后的 x（可 host 完成）。
- `get_inputs()`：复用 base（含 `torch.manual_seed(42)`），`x_input_coords` `(256,3)`，mask 全 1 `(256,)`。
- `get_init_inputs()`：返回 `[4, 1.0, False]`。
- **无 state_dict 匹配问题**：base 无 buffer/parameter（n_sample/s_trans/centre_only 是 Python 属性）。

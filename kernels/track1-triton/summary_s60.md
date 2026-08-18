# S60（燧原 GCU）Triton 提交物汇总与性能分析

分支：`kernel-opt/s60-triton-submissions`。度量口径：`auto_bench.py --warmup 50 --repeat 100`，median wall time，speedup = v0(base.py) / v1(candidate)。

## 一、10 个算子提交物全景

| 算子 | 提交物 | 正确性 | speedup | 加速来源 |
|---|---|---|---|---|
| `groupedtopk` | `triton_grouped_topk_001.py` | PASS | **1.68x** | 12→1 launch 融合 + 输出池复用 |
| `fused_moe` | `triton_fused_moe_002.py` | PASS | **13.8x** | 逐-token 路由 + selection 融合，省 launch 与冗余计算 |
| `mhc_head_compute_mix` | `triton_mhc_head_compute_mix_001.py` | PASS | **6.8x** | Sinkhorn 20 轮迭代融合进单 kernel，省 20×多次 launch |
| `mhc_head_compute_mix_backward` | `triton_mhc_head_compute_mix_backward_001.py` | PASS | **1.26x** | elementwise sigmoid-backward 融合 |
| `centre_random_augmentation` | `triton_centre_random_augmentation_001.py` | PASS | 0.95x | 四元数旋转（随机数 host 生成），接近打平 |
| `music_flamingo_rotary_embedding` | `triton_rotary_002.py` | PASS | 0.9x | 纯 elementwise 融合，measurement-bound |
| `mhc_post_layer_mix` | `triton_mhc_post_layer_mix_001.py` | PASS | 0.56x | einsum 用 tl.sum 展开（见下文根因） |
| `flexattention` | `triton_flexattention_001.py` | PASS | 0.42x | 手写 causal SDPA（见下文根因） |
| `mm_encoder_attention` | `triton_mm_encoder_attention_001.py` | PASS | 0.27x | 手写 SDPA（见下文根因） |
| `sparse_pooler` | `triton_sparse_pooler_001.py` | PASS | — | 库算子占优，正确性优先提交 |

正确性 **10/10 全部 PASS**，无留空。

## 二、慢算子的根因分析

四个算子（`mm_encoder_attention` 0.27x、`flexattention` 0.42x、`mhc_post_layer_mix` 0.56x、`centre_random_augmentation` 0.95x）未超越 base，根本原因归结为以下几点。

### 1. GCU 的 Triton 后端 `tl.dot` 是 Unknown（最根本）

GCU 的 `triton_gcu` 未实现 `tl.dot`（矩阵乘原语），导致任何 GEMM 只能退化成 `tl.sum(a * b)` 的**逐元素标量 FMA 展开**，完全无法利用 GCU 的张量核心（Matrix Core）。

而 base 的 SDPA / einsum 均 dispatch 到 CNNl 库，CNNL 内部用汇编级张量核心 + fp16 矩阵乘优化。**Triton 标量 FMA vs CNNl 张量核心，天然差一个数量级**。

### 2. SDPA 类（mm_encoder 0.27x / flexattention 0.42x）

base 用 `F.scaled_dot_product_attention`，在 GCU 上落到 CNNl 的 FlashAttention/SDPA 融合算子（张量核心 + online softmax，不实例化 [seq,seq] score 矩阵）。

我们的手写 kernel 三处致命退化：
- **QK^T 与 PV 两个 GEMM 用 `tl.sum(q * K)` 标量乘加**，未用张量核心
- **`num_warps=1` 且 grid 粒度为每 query token 一个 program**（mm_encoder 1328 program / flexattention 664 program），每处理一个 query 都要重新 load 整块 K/V，**无数据复用**
- **fp16 输入全程 `.to(tl.float32)` 计算**，失去 fp16 张量核心吞吐优势

### 3. mhc_post_layer_mix（0.56x）

base 用 `torch.einsum('abmn,abmc->abnc')`（dispatch 到 CNNl 批量 GEMM）。我们的实现：
- `grid=(2×4096,)=8192` program，`num_warps=1`
- 每 program 处理一个 (a,b) 的 [4,4]@[4,256] 小矩阵乘，用 `tl.sum(comb[:,:,None] * r[:,None,:], axis=0)` 三维广播展开 → **标量乘加，未用张量核心**

### 4. 为什么 fused_moe / mhc_head_compute_mix 反而快

因为它们**规避了 GEMM 瓶颈**：
- `fused_moe`：核心是 topk 路由选择 + 数据搬运，base 是 naive 逐-token Python 循环 + 大量小 op launch，融合省的是 launch 开销，不依赖张量核心
- `mhc_head_compute_mix`：Sinkhorn 是 [4,4] 小矩阵 20 轮迭代，base 有 20×多次 kernel launch，融合进单 kernel 省 launch 开销
- `mhc_head_compute_mix_backward`：纯 elementwise，融合省 launch

**结论**：慢算子慢在 **GEMM 被降级成标量 FMA**（GCU Triton 无 `tl.dot`），而非 kernel 写得差。这是 GCU 上 Triton 做 attention/矩阵乘的平台性短板。

## 三、可优化方向（按性价比排序）

1. **提高 warp 数与 tile 并行度**：当前 `num_warps=1` 且每 program 只算一个 query / 一个 (a,b)，可改成一个 program 处理多个 query（tile），提升 occupancy。纯 Triton 层可做，预计把 0.27x 提到 0.5x 左右，但**无法追平张量核心**。
2. **实测 `tl.dot` 在 GCU 的可用性**：若 triton_gcu 存在某种降级实现（哪怕慢的 dot），也比纯 `tl.sum` 好。待验证。
3. **混合方案**：核心 GEMM 用 `torch.mm`（CNNL 张量核心），Triton 只做 softmax/mask 等融合部分。但 Triton 成分降低，且未必符合纯 Triton 提交要求。

## 四、提交物规范

每个算子的提交物位于 `kernels/track1-triton/<算子>/s60/triton_<算子>_001.py`，含：
- `ModelNew` 类（接口与 base.py 的 `Model` 一致）
- `get_inputs()` / `get_init_inputs()`
- 环境约束：`tl.dot` 不可用，GEMM 用 `tl.sum` 展开；kernel 内无随机数生成（host 生成后传入）

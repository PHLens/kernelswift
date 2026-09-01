# `index_topk`（Ascend 910B4）优化总结

## 当前结果

| 实现 | 正确性 | Median latency | 相对基线 |
|---|---|---:|---:|
| `base.py` | reference | `8.790610 ms` | `1.000x` |
| `ascendc/candidate_001.py` | PASS | `8.524795 ms` | `1.033x` |
| `ascendc/candidate_004.py` | PASS | `8.349305 ms` | **`1.053x`** |
| `ascendc/candidate_005.py` | PASS | `8.326480 ms`（不同 50-repeat run） | `1.052x` |
| `ascendc/candidate_006.py` | PASS | `14.463780 ms`（短 paired benchmark） | `0.608x`（rejected） |
| `ascendc/candidate_007.py` | PASS | `8.298150 ms`（稳定复测） | **`1.059x`** |
| `ascendc/candidate_008.py` | PASS | `10.657250 ms` | `0.827x`（rejected） |

`ascendc/candidate_004.py` 曾是 canonical candidate。在 `warmup=20, repeat=100` 的稳定复测中，它将完整 forward 中位延迟从 `8.790610 ms` 降至 `8.349305 ms`，提升约 `5.02%`，达到 kernel-opt-loop 默认 `5%` adoption threshold。

当前 canonical candidate 为 `ascendc/candidate_007.py`。首次 paired run 为 `8.801935 ms → 8.295765 ms`，speedup `1.061x`；第二次稳定复测为 `8.784070 ms → 8.298150 ms`，speedup `1.059x`。两次测试 accuracy 均为 PASS。

## 已采用优化

1. 在模型构造期预计算固定 causal mask 和每 token 有效 candidate 数，删除 forward 内的 `arange`、`repeat`、比较、整除和 mask 构造。
2. 将 ReLU 后的按头权重乘法改为原地执行，减少中间张量和一次大张量写回。
3. 对固定 scope 的 `offset=0` 增加精确分支，删除 top-k 输出上的无效逐元素加法。
4. 保留原生 `torch.topk`，维持与 reference 相同的相等分数索引选择语义。
5. `candidate_007` 将构造期 causal mask 保存为 BF16。mask 仅包含 BF16 可精确表示的 `0` 和 `-inf`，与 BF16 `scores` 同 dtype，可避免 forward 中 mask add 后的额外 dtype cast，并减少 mask 读取量。

## 后续候选状态

- `candidate_006.py`：自定义 post-BMM 路径的 `.so` 已成功构建，独立 exact probe 的 Phase A（BF16 product 位级比较）和 Phase B（最终 indices）均全等，完整短 paired benchmark 的 accuracy 也为 PASS。但 paired latency 为 `8.788365 ms → 14.463780 ms`（`0.608x`），性能显著回退，因此 benchmark rejected，不作为 canonical。构建阶段曾因陈旧的 CMake `ExternalProject` stamp 误判目标已完成，清理该 stamp 后才重新执行真实构建；随后修复了设备侧同步以及 `DataCopyPad` 搬运问题，才获得可加载 `.so` 和 exact PASS。现有实现替换 `ReLU + head-weight multiply`，head reduction 仍由后续 `sum(dim=2)` 完成。
- `candidate_007.py`：accuracy PASS，现为 canonical。首次 paired run 为 `8.801935 ms → 8.295765 ms`（`1.061x`），第二次稳定复测为 `8.784070 ms → 8.298150 ms`（`1.059x`）。
- `candidate_008.py`：在 `candidate_007` 上仅将固定 `start_pos=0, offset=0` 分支的 `torch.where(indices >= valid, -1, indices)` 改为等价的原地 `indices.masked_fill_(indices >= valid, -1)`。accuracy PASS，但 paired latency 为 `8.812960 ms → 10.657250 ms`（`0.827x`），出现明显性能回退，因此 rejected。当前 Ascend lowering 下，该原地写法没有产生有利的融合或内存节省。

## Profiler 结论

`candidate_001` 的主要每-forward 设备热点为：

| 热点 | 时间 |
|---|---:|
| BatchMatMul | `2.2411 ms` |
| InplaceMul | `1.9925 ms` |
| InplaceRelu | `1.3306 ms` |
| ReduceSum | `0.9622 ms` |
| TopK | `0.6010 ms` |

当前剩余主要空间是融合 `ReLU + head-weight multiply + head reduction`。完整 `torch.compile` 实验在当前 Ascend 栈触发 vector-core runtime error；仅编译该局部链虽然可运行，但改变浮点归约结果，最终 top-k 索引不精确，二者均未保留为候选。

## 优化难点与空间判断

`index_topk` 并非理论上没有优化空间，而是剩余空间集中在难以同时满足性能与精确语义的核心链路。固定输入下，BMM 生成的 `scores` 形状为 `[8,2600,16,650]`，包含约 `2.16` 亿个 BF16 元素，单份约 `432 MB`。后续 `ReLU`、按 head 乘权、head reduction、mask 和 top-k 会对该数据产生多次读写；其中 profiler 中 `InplaceRelu + InplaceMul + ReduceSum` 合计约 `4.2853 ms`，因此真正有价值的方向是三者融合，并直接输出 `[8,2600,650]`，避免完整大张量写回和后续重读。

当前实验尚未兑现这部分理论空间，原因如下：

1. `candidate_006` 只用 Ascend C 替换了 `ReLU + head-weight multiply`，仍输出完整 `[8,2600,16,650]`，head reduction 继续由独立的 `sum(dim=2)` 完成，因此没有消除最关键的大张量写回、重读和 reduction kernel launch。
2. 为精确复刻原路径的 BF16 行为，该 kernel 对每个 head 执行 BF16→FP32 cast、FP32 ReLU/Mul、RNE cast 回 BF16及数据搬运，并串行处理 16 个 heads、进行多次 event 同步。这些额外成本使 exact PASS 的实现反而退化至 `0.608x`。
3. 真正融合 head reduction 会改变并行归约顺序、累加精度或中间舍入位置。即使浮点值近似相等，也可能改变最终 top-k 索引；已有局部编译实验已观察到此问题，所以不能仅以普通数值容差放行。
4. 自定义 top-k 还必须复刻原生 `torch.topk` 的 tie 选择语义。masked-topk 实验的 `681` 个差异发生在相等分数上，说明排序数值正确仍不足以保证索引全等，因此当前 canonical 必须保留原生 top-k。
5. 已采用的 mask 预计算、原地乘法、`offset=0` 快路径和 BF16 mask 都属于不改变归约及 top-k 语义的安全外围优化；这些优化已取得稳定 `1.059x`，但无法消除核心 scores 链路的数据流量。

因此，当前结果应理解为“容易且语义安全的空间已基本兑现”，而不是“算子没有理论空间”。下一步最有价值的实验是实现严格复刻 BF16 中间舍入和原生 head 归约语义的 `ReLU + multiply + 16-head reduction` 融合 kernel，直接产生 `[8,2600,650]`；应先通过独立 product/reduction 位级验证和最终 indices exact probe，再评估端到端性能。继续微调 Python 表达式或仅替换单个逐元素算子，预期收益有限。

## Ascend C exact-probe 状态

完整 forward exact probe 已建立并可运行。`candidate_006` 的 post-BMM child kernel 已通过独立 Phase A BF16 product 位级全等、Phase B indices 全等以及完整短 benchmark accuracy PASS，但因 `0.608x` 的显著性能回退被拒绝。另一条自定义 masked-topk child kernel 仍有 `681` 个索引差异；首个差异的实际分数与期望分数完全相等，说明问题是 `torch.topk` 的 tie 选择语义，而不是数值排序错误。masked-topk 完整自定义 forward 因此仍保持 `Unknown`。当前 canonical `candidate_007` 不调用上述 child kernel。

## 复现命令

```bash
/usr/local/python3.11.15/bin/python3 auto_bench.py \
  --v0_file kernels/track2-clike/index_topk/base.py \
  --v1_file kernels/track2-clike/index_topk/ascendc/candidate_007.py \
  --warmup 20 --repeat 100 --fail-fast
```

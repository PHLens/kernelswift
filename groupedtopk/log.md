# Grouped Top-K Triton Kernel Optimization Log

本文记录 `base.py` 中 grouped top-k 在 MLU590-H8 上的 Triton 优化过程。每次优化独立成 entry，记录当时现状、假设、优化手段、踩坑、结果、与性能上界的差距，以及下一步方向。

## 1. 固定问题与测试口径

### 1.1 算子语义

- 输入：`gating_output: float32[T, 256]`
- 当前核心 shape：`T=83`
- expert：256
- group：8，每组 32 个 expert
- group top-k：4
- expert top-k：8
- scoring：softmax
- 输出权重在选中 top-8 上重新归一化
- `routed_scaling_factor=1.0`
- 输出：`weights: float32[T, 8]`，`ids: int32[T, 8]`

softmax 对所有 expert 是单调变换，而最终只在选中的 8 个 expert 上重新归一化，因此 Triton kernel 可以先直接在 logits 上完成 group top-k 和 expert top-k，最后仅对选中的 logits 做 softmax。这避免了对 256 个 expert 做完整 softmax，同时保持当前配置的数学等价性。

### 1.2 环境

- Device：MLU590-H8
- 可见 MLU core：48
- PyTorch：`2.11.0+cpu`
- torch_mlu：`1.32.0+torch2.11.0`
- Triton：`3.2.0`
- Python：`/projs/framework/lipenghui/venv/pytorch_main/bin/python`

### 1.3 测量规则

1. kernel device time 以 profiler JSON 中 `cat == "kernel"` 的 `dur` 为准，单位为微秒。
2. 最终 Triton 对比使用预分配输出、20 次 warmup、50 次稳态调用。
3. wall time 使用同步包围的循环平均值；最终数据采用 baseline/optimized 交错顺序、每轮 500 次、9 个 repeat 的中位数。
4. 不把单个 kernel 的 device time、整段 device span、host wall time混在一起比较。
5. 早期 quick benchmark 受共享机器 host 负载影响较大，只用于淘汰明显退化方案；是否保留优化以同一 profiler trace 内的 device time为准。
6. 正确性要求：非 tie 随机输入的 ID 完全一致，权重 `allclose`；另外覆盖递增、递减、重复值、全相等和 scaling factor。

历史上出现过不能由当前 JSON 重现的数字，不作为本日志的正式结论。主表只使用现存 profiler JSON 或最终交错 benchmark 的结果。

## 2. Upbound 定义

这里的 upbound 指“可达到的性能上界”，等价地对应“延迟下界”。目前有两个层次：

1. **实测工程 upbound**：`torch_mlu_ops.moe_softmax_topk` 的 TMO/BANGC fused kernel，在同一 MLU590-H8、同一 shape、预分配输出下为 **9.9656 us**。
2. **定制 BANGC 估算**：针对 `[83, 256]` 完全固定语义的专用 BANGC kernel，估计可能达到 **7-9 us**。该范围没有自写 BANGC 实测，只能作为 stretch goal，不能当成正式基线。

TMO kernel 不是 Triton 实现。二进制符号和 profiler kernel 名均为 `tmo::kernels::MLUSoftmaxTopkKernel<float, float>`；其 launch shape 为 `dim=(4, 12, 1)`，说明每个 cluster 使用 4 个 core，共 12 个 cluster。对应 trace：

- [TMO 50-iteration trace](log/tmo_moe_softmax_topk_T83_preallocated_50iter.pt.trace.json)

## 3. 当前结果总览

| 实现 | Kernel 数 | Device time | 相对上一阶段 | 相对 TMO |
|---|---:|---:|---:|---:|
| `torch.compile(mode="reduce-overhead")` 图 | 17 | 112.4800 us | - | 11.29x |
| 首版 fused Triton | 1 | 22.1136 us | 5.09x faster | 2.22x |
| 当前 T=83 optimized Triton | 1 | **20.1264 us** | 8.99% lower | **2.02x** |
| TMO/BANGC reference | 1 | **9.9656 us** | - | 1.00x |

当前 Triton 相对 compiled PyTorch 图的 device speedup 为 **5.59x**。从 compiled PyTorch 的 112.48 us 到 TMO 的 9.9656 us 这段可优化区间看，当前 Triton 已消除约 **90.09%** 的延迟差距。

但从当前 Triton 自身看，距离 TMO 仍有 **10.1608 us** 绝对差距：

- 当前 Triton latency / TMO latency：**2.0196x**
- 当前 Triton 等效吞吐 / TMO 吞吐：**49.52%**
- 若以 TMO 为目标，当前 latency 仍需再下降 **50.48%**

最终交错 wall-time 长测：

| 实现 | Wall time/call | 结果 |
|---|---:|---:|
| 首版 Triton preallocated launcher | 31.181 us | baseline |
| 当前 T=83 三参数专用 launcher | **27.253 us** | **下降 12.60%** |

## 4. Optimization Entries

### Entry 000 - PyTorch eager 与 compile 起点

**状态**

`base.py` 由 softmax、group max、两次 top-k、mask/scatter、除法和类型转换等多个 PyTorch op 组成。

**优化手段**

启用 `torch.compile(mode="reduce-overhead")`，主要收益来自减少 host launch 开销并融合部分 pointwise op。

**踩坑**

- compile 后仍保留 CNNL top-k、scatter、reduce 等 library kernels，无法完整融合成一个 kernel。
- kernel time 总和和整段 device span不是同一指标。
- eager 中 host launch gap 很大；仅查看 kernel time 会低估端到端问题。

**结果**

- compiled trace：17 kernels，kernel time 总和 **112.4800 us**。
- 典型 kernel 包括 2 个 top-k、scatter、softmax、reduce、cast 和若干 Triton pointwise kernel。
- 对应 trace：[compiled PyTorch trace](log/bjysw0102_2816309.1785405608465243461.pt.trace.json)

**与 upbound 的差距**

- 相对 TMO：`112.48 / 9.9656 = 11.29x`。
- 结论：继续依赖通用 PyTorch/CNNL op 组合无法逼近单 fused kernel 的目标，必须融合完整路由逻辑。

**下一步**

实现单 Triton kernel，将 group selection、expert selection、归一化和输出写回全部融合。

---

### Entry 001 - 首版单 fused Triton kernel

**状态**

compiled PyTorch 图仍有 17 个 device kernels 和多次 host launch。

**优化手段**

新建 [triton_grouped_topk.py](triton_grouped_topk.py)，不修改 `base.py`：

- 一个 persistent Triton kernel，最多 launch 48 个 program。
- 每个 program 按 `program_id, program_id + grid, ...` 处理 token。
- 一次加载 256 个 logits。
- `[8, 32]` group max。
- 4 次串行 group argmax，选 top-4 group。
- mask 未选 group 后，8 次串行 256-lane expert argmax。
- 只对选中的 8 个 logits 做归一化。
- 使用 `selected_rank[256]` 让 8 个选中 lane并行 scatter 到紧凑输出。

**踩坑**

- MLU Triton 没有可直接返回 value/index pair 的 `argsort`；`tl.sort` 只有 value，无法直接维护 expert ID。
- top-k 使用重复 argmax，因此仍有 4 + 8 轮串行 reduction。
- 编译后的 MLISA 约使用 **878 GPR** 和 **18.5 KB NRAM**，状态量较大。

**结果**

- 50 次稳态 kernel 平均：**22.1136 us**。
- min/max：21.76 / 22.56 us。
- launch：`dim=(48, 1, 1)`，U1，12/12 clusters。
- 相对 compiled PyTorch：**5.09x faster**。
- trace：[first Triton 50-iteration trace](log/triton_grouped_topk_T83_preallocated_50iter.pt.trace.json)

**正确性**

- 随机 `T=1/7/48/83/97` 通过。
- IDs 与 `base.py` 一致，随机权重最大误差不超过 `5.96e-8`。
- 覆盖递增、递减、重复值、全相等和 scaling factor。

**与 upbound 的差距**

- 相对 TMO：`22.1136 / 9.9656 = 2.219x`。
- 等效吞吐为 TMO 的 **45.07%**。
- 还剩 **12.1480 us** 绝对 device 差距。

**下一步**

首先分离 allocator/launcher 成本，再针对 12 轮串行 argmax 做算法和 lowering 优化。

---

### Entry 002 - 预分配输出与 fast launcher

**状态**

首版 API 每次调用创建 `weights` 和 `ids`，wall time 中包含 allocator 和 Python wrapper 开销。

**优化手段**

- 增加 `grouped_topk_triton_out`，允许调用方预分配输出。
- 使用 `triton.runtime.fast_libentry` 缩短 launcher 路径。
- 另提供 `torch.library.triton_op` 入口以便嵌入更大 `torch.compile` 图。

**踩坑**

- 这是 host/allocator 优化，不会降低 kernel device duration。
- allocated、preallocated、compiled wrapper 必须分开报告。
- 共享机器 host 负载会让短调用 wall time明显漂移。

**结果**

- 历史同轮 wall benchmark：allocated 约 **79.6 us**，preallocated 约 **30.5 us**。
- device kernel 仍约 **22.1 us**。
- 结论：研究 kernel 时必须使用预分配输出，否则 allocator 会掩盖 device 优化。

**与 upbound 的差距**

- device gap 不变，仍为 TMO 的约 2.22x。
- wall gap明显缩小，但 TMO preallocated wall 实测约 10.56 us，仍有较大 launcher 和 device 双重差距。

**下一步**

固定预分配口径，所有后续候选优先比较 profiler device time。

---

### Entry 003 - `num_warps` / `num_stages` 与 grid 探索

**状态**

需要确认简单 launch/meta 参数是否能改善资源占用或并行度。

**优化手段**

- 探索 `num_warps`、`num_stages`。
- 探索不同 program 数；当前设备有 48 个 core。

**踩坑**

- 部分配置编译超过 6 分钟，编译成本明显异常。
- quick benchmark 的约 0.2 us 波动不足以超过测量噪声。
- program 数高于物理 core 不等于更高并行度；`T=83` 时 grid=48 已自然让部分 program 处理第二个 token。

**结果**

- `num_warps/num_stages` 没有得到可稳定复现的收益，最终保留 `1/1`。
- 后续 group-rank 候选的同 trace grid 数据进一步确认：grid=42 为 21.0528 us，grid=48 为 **20.2456 us**，grid=83 为 22.6096 us。
- 结论：48 program 最适合当前 shape。

**与 upbound 的差距**

简单 meta tuning 不足以解释与 TMO 约 2x 的差距，必须改变 reduction 结构或 core 协作方式。

**下一步**

减少 256-lane状态或串行 argmax 次数。

---

### Entry 004 - 去掉 `selected_rank[256]`，直接逐 rank 写输出（失败）

**状态**

首版 kernel 的 `selected_rank[256]` 占用较多 NRAM/GPR，因此尝试直接把每轮 argmax 的标量结果写入输出。

**优化手段**

- 每次 expert argmax 后直接写 `weights[row, rank]` 和 `ids[row, rank]`。
- 8 轮结束后，从输出重新加载 8 个值并归一化。

**踩坑**

- MLU Triton 对循环内标量 store/reload 的 lowering 很差。
- 虽然资源下降，但 GDRAM 往返和标量控制路径远比预期昂贵。
- “减少 NRAM”不自动等价于“降低 latency”。

**结果**

- MLISA 资源从约 878 GPR / 18.5 KB NRAM 降到约 **630 GPR / 4.9 KB NRAM**。
- quick wall benchmark 最好仍约 **83-84 us**，显著慢于约 31 us baseline。
- 方案淘汰，不进入最终代码。

**与 upbound 的差距**

明显退化，无需继续与 TMO 定量比较。

**下一步**

保留 lane-parallel masked store，避免循环内标量 GDRAM 通信。

---

### Entry 005 - 将 256 expert 压缩为 128 candidates（失败）

**状态**

选中 4 个 group 后只有 128 个有效 expert，但首版 expert top-8 仍在 256 lane 上 reduction，其中一半是 `-inf`。

**优化手段**

尝试三种 compact 方式：

1. 根据动态 group ID 构造 128 个 expert ID，再做动态 gather。
2. 对四个连续 32-element group分别 load，再用 nested `tl.join` 拼成 128 candidates。
3. 在 128 candidates 上保留 `selected_rank` 并做 masked scatter。

**踩坑**

- `tl.cat` 当前实现要求 `can_reorder=True`，不能保证需要的 ID 顺序。
- 动态 expert ID load 被 lowering 为通用 `linalg_ext.gather`，抵消 reduction 宽度减半的收益。
- `tl.join` 虽避免 gather，但引入 4 次额外 32-element GDRAM load 和大量 memref copy/interleave。
- compact 后还需维护 candidate position 到原 expert ID 的映射。

**结果**

- dynamic gather `compact_rank` wall 约 32.1 us，对比同轮 baseline 31.3 us，没有收益。
- four-load + join 的 profiler device time：**24.2072 us**。
- 同 trace baseline：22.1160 us，即 device latency **增加 9.45%**。
- 候选 trace：[candidate comparison trace](log/triton_grouped_topk_optimized_candidates_T83_50iter.pt.trace.json)
- 方案淘汰。

**与 upbound 的差距**

compact-join 相对 TMO 为 2.43x，比首版 Triton 更差。

**下一步**

若再次尝试 compact，必须在第一次 256-element load 后仅通过 NRAM layout变换完成，不能重新从 GDRAM gather/load。需要先确认 MLU Triton 是否提供可控的 vector permute/layout primitive。

---

### Entry 006 - 关闭 argmax stable tie-break（严重失败）

**状态**

默认 `tl.max(..., return_indices=True)` 使用 `return_indices_tie_break_left=True`。MLISA 中每次 argmax 后有额外 tie 修正逻辑，因此尝试关闭稳定 tie-break。

**优化手段**

对 group 和 expert argmax 设置 `return_indices_tie_break_left=False`。

**踩坑**

- 当前 MLU Triton 后端只有默认 tie-break 路径能匹配专用 `argmax.nan/value.nram` lowering。
- 关闭 tie-break 后反而生成通用 `linalg.reduce`，且 reduction 形状变为 7/255 lanes。
- GPR 增长到约 **1414**，MLISA 从约 1140 行增至约 1819 行。

**结果**

- grid=48 quick wall time约 **508.6 us**，baseline 约 30.95 us。
- 方案严重退化并淘汰。

**与 upbound 的差距**

该失败属于 compiler lowering 问题，不是算法上限。

**下一步**

继续使用默认 indexed argmax。后续任何 reduction API 改写，都必须检查 `.linalgopt` 和 `.mlisa` 是否仍命中专用 NRAM 指令。

---

### Entry 007 - 只在 8 个输出值上归一化（失败）

**状态**

首版最终在 256 lanes 上构造 sparse numerator 并做 max/sum，理论上只需对 8 个值归一化。

**优化手段**

- 每轮 expert argmax 将标量 value/id收集进 8-lane tensor。
- 最后只在 8 lanes 上计算 exp/sum。

**踩坑**

- Triton 源码中的“8-lane tensor”没有被当前 MLU 后端 lowering 成廉价寄存器收集。
- 循环中的标量插入和状态更新产生高成本控制/拷贝路径。
- 算术量减少，但数据组织成本大幅增加。

**结果**

- quick wall time约 **61.34 us**，同轮 baseline 31.96 us。
- 方案淘汰。

**与 upbound 的差距**

明显退化。当前 256-lane sparse normalization 虽然看似浪费，但后端 vector lowering 更高效。

**下一步**

除非能使用后端原生 top-k/value-index收集 primitive，否则保留 sparse lane-parallel normalization。

---

### Entry 008 - 并行 group rank 替换 4 次串行 group argmax（成功）

**状态**

首版 group top-4 使用 4 次串行 8-lane indexed argmax。虽然 group 数只有 8，但每次 indexed argmax 都包含 reduction 和 tie 处理。

**优化手段**

新建 [triton_grouped_topk_optimized.py](triton_grouped_topk_optimized.py)：

- 构造 `[8, 8]` group score比较矩阵。
- 对每个 group 统计有多少 group 的 score更大。
- 相等时用更小 group ID作为确定性 tie-break。
- `rank < 4` 即选中 group。
- expert top-8、sparse normalization 和 masked scatter 保持已知最优路径。

该改写用一次并行 compare/rank 替代 4 次串行 indexed argmax。

**踩坑**

- wall quick benchmark 有时显示持平或略慢，不能据此否定；必须看同一 profiler trace。
- grid 需要重新测量，不能假设 83 program 更好。

**结果**

同一 candidate trace 中：

| Variant | Grid | Device time |
|---|---:|---:|
| 首版 Triton | 48 | 22.1160 us |
| group-rank | 42 | 21.0528 us |
| group-rank | 48 | **20.2456 us** |
| group-rank | 83 | 22.6096 us |

- grid=48 相对首版 device latency下降 **8.46%**。
- 正确性覆盖随机和 tie case，ID 完全一致。
- 方案保留。

**与 upbound 的差距**

- `20.2456 / 9.9656 = 2.032x`。
- 相对首版已缩小约 1.87 us，但 expert top-8 的 8 次 256-lane串行 argmax 成为更明显的主瓶颈。

**下一步**

固定 `T=83`、stride 和 scaling，减少动态参数与循环边界；同时寻找 expert top-8 的层次化或 selection-network实现。

---

### Entry 009 - T=83 compile-time 特化，但通用 launcher 变慢（部分成功）

**状态**

group-rank kernel 仍接收动态 `num_tokens`、row stride、scaling factor 和多个 constexpr meta参数。

**优化手段**

- 将 `T=83`、row stride `256`、scaling `1.0` 作为 compile-time常量。
- 去掉动态循环边界和最终 scaling 乘法。

**踩坑**

- 首次实现仍复用通用 Python entry，并额外传递 4 个 constexpr控制参数。
- device time下降，但 `fast_libentry` 的 host参数解析成本增加。
- 这是典型的“kernel 更快、算子 wall time反而更慢”。

**结果**

- device：约 **20.1456 us**，同 trace baseline 22.0864 us。
- 但交错 wall：baseline 31.209 us，特化入口约 **33.639 us**，反而慢约 7.8%。
- 特化算法有效，但 launcher 设计失败，不能作为最终入口。

**与 upbound 的差距**

- device 已接近 20.1 us，但端到端结果不合格。

**下一步**

建立独立的固定 T=83 kernel signature，只保留 `logits/weights/ids` 三个指针参数，不在运行时传任何 shape/meta/scaling参数。

---

### Entry 010 - T=83 三参数专用 kernel/launcher（当前最佳）

**状态**

Entry 009 已证明 compile-time特化能降低 device time，但通用 entry 的参数解析抵消收益。

**优化手段**

- 新增 `_grouped_topk_group_rank_t83_kernel(logits_ptr, weights_ptr, ids_ptr)`。
- kernel 内硬编码 `T=83`、`E=256`、`group=8x32`、`topk_group=4`、`topk=8`、`scaling=1.0`。
- launcher 仅传三个 device pointer。
- 仍使用 grid=48、`num_warps=1`、`num_stages=1`。
- Python API 在满足连续 `[83,256]` 且 scaling=1.0 时自动选择专用 entry，其他输入回退通用 group-rank kernel。

**踩坑**

- 专用 entry 会增加代码重复，但这是消除 Python/Triton launcher meta参数开销所必需的特化。
- 若 shape 或 scaling 不匹配，不能错误进入专用 kernel；wrapper 必须严格 guard。
- profiler trace 中同时包含 50 个 baseline 和 50 个 optimized kernel，不能直接对全部 100 个 event 求平均，必须按 kernel name分组。

**结果**

最终同 trace device 数据：

| Kernel | Count | Average | Min | Max |
|---|---:|---:|---:|---:|
| `_grouped_topk_softmax_kernel` | 50 | 22.0872 us | 21.6000 us | 22.6800 us |
| `_grouped_topk_group_rank_t83_kernel` | 50 | **20.1264 us** | 19.7200 us | 20.6400 us |

- device latency下降 **8.88%**。
- device speedup：**1.0974x**。
- 最终交错 wall：31.181 us -> **27.253 us**。
- wall latency下降 **12.60%**，wall speedup **1.144x**。
- 当前 trace：[final optimized trace](log/triton_grouped_topk_group_rank_fixed_T83_preallocated_50iter.pt.trace.json)
- 当前 benchmark：[benchmark_triton_grouped_topk_optimized.py](benchmark_triton_grouped_topk_optimized.py)
- profiler复现：[profile_triton_grouped_topk_optimized.py](profile_triton_grouped_topk_optimized.py)

**正确性**

- 随机 `T=1/7/48/83/97`：IDs 完全一致。
- 递增、递减、重复值、全相等：IDs 完全一致。
- scaling=2.5 走通用路径并通过。
- 最大权重误差：`1.192e-7`；固定 T=83 edge回归最大误差 `5.96e-8`。

**与 upbound 的差距**

- 相对 TMO：`20.1264 / 9.9656 = 2.0196x`。
- 等效吞吐为 TMO 的 **49.52%**。
- 绝对差距：**10.1608 us**。
- 以当前 Triton 自身为基数，要达到 TMO latency 仍需再下降 **50.48%**。
- 从首版 Triton 到 TMO 的剩余区间中，本轮优化消除了约 **16.36%**。

**结论**

当前优化是稳定但增量式的成功。它解决了 group top-4 和 launcher 的一部分成本，但未触及最大的 expert top-8 reduction差距，因此不能期待仅靠继续微调 meta参数逼近 10 us。

---

### Entry 011 - 分层 expert top-8 算法设计（分析完成，未实现）

**状态**

当前最佳 kernel 在 group top-4 之后保留 256 lanes，其中只有 4 个 group、共 128 个 expert 有效；随后执行 8 次串行 256-lane indexed argmax。按二叉 reduction tree 粗略建模，expert selection 部分约为：

- 比较量：`8 * (256 - 1) = 2040` 次。
- reduction 串行深度：`8 * log2(256) = 64` 层。
- winner mask/update：`8 * 256 = 2048` lane 次。

本 entry 只分析精确分层 top-k 的算法空间，不修改 Triton kernel，因此沿用 Entry 010 的 **20.1264 us** device baseline，不产生新的 profiler 数据。

**算法不变量与正确性**

group top-4 选中 4 个 32-expert group。对每个选中 group 只保留 local top-8 足以恢复最终 global top-8：若某 expert 在本 group 内排在第 9 名或之后，则同一已选 group 内至少已有 8 个更优 expert，它不可能进入 4 个 group 合并后的 global top-8。

所有候选必须携带 `(value, global_expert_id)`，比较顺序与当前实现保持一致：value 降序，value 相等时按既有 expert ID tie-break。softmax 和最终归一化不变，只替换 expert selection。

**候选 A：local top-8 + 32-candidate flat merge**

1. 将选中的 4 个 group 理想地 compact 为 `[4, 32]`。
2. 4 个 group 并行执行 8 轮 32-lane argmax，得到 `[4, 8]`。
3. 将 32 个候选展平，再执行 8 轮 32-lane argmax。

不复用已有 group max 时，理想比较量为 `4 * 8 * 31 + 8 * 31 = 1240`，串行深度为 `8 * 5 + 8 * 5 = 80`。若 group max 同时返回 local ID 并复用为各组 local top-1，则比较量降为 `4 * 7 * 31 + 8 * 31 = 1116`，比当前 2040 下降 **45.29%**；串行深度仍有 `7 * 5 + 8 * 5 = 75`，高于当前的 64。

结论：该方案减少总工作量，但形成“local selection 完成后才能 global selection”的长依赖链，不是首选。

**候选 B：local sorted top-8 + 4-way head merge（首选分层结构）**

1. 先得到 group max 及对应 local ID，并选出 4 个 group。
2. compact 为 `[4, 32]`，4 个 group 并行生成各自有序的 local top-8 list。
3. 维护 4 个 list cursor；每轮只比较 4 个 list head，输出 winner 并推进对应 cursor，共执行 8 轮。

若 local list 仍用重复 argmax产生，并复用已有 local top-1，则 expert selection 的新增比较量约为：

- local top-2 到 top-8：`4 * 7 * 31 = 868`。
- 4-way merge：`8 * (4 - 1) = 24`。
- 合计：`892`，相对当前 2040 理论下降 **56.27%**。

对应串行深度约为 `7 * log2(32) + 8 * log2(4) = 51` 层，相对当前 64 层下降 **20.31%**。它同时减少 reduction 宽度、总比较量和 mask/update 工作量，是最保守且算法收益明确的分层方案。

**候选 C：local partial sorting network + 4-way head merge（高收益候选）**

用 compare-swap network 同时维护 value 和 ID，为 4 个选中 group 并行生成有序 local top-8。作为保守上界，完整 bitonic sort-32 需要 15 个 compare-swap stage、每组 240 个 comparator；4 组共 960 个 comparator。再加 4-way merge 的 24 次比较：

- 总比较量上界约为 `960 + 24 = 984`，仍比当前下降 **51.76%**。
- 串行深度约为 `15 + 8 * 2 = 31` 层，比当前 64 层下降 **51.56%**。
- 只生成 top-8 的 partial network 可以进一步减少 comparator 数，但必须单独验证网络正确性。

从纯算法深度看，该方案最有机会接近 TMO；代价是 compare-swap IR 规模、value/index pair 搬运和编译复杂度更高。

**备选：compact-128 partial top-k network**

也可以把 4 个选中 group直接 compact 为 128 candidates，再构造 top-8 partial network。完整 bitonic sort-128 的比较量约为 1792、深度为 28，工作量只比当前重复 argmax 下降约 12%，但依赖深度显著下降。partial network 会更优；不过它放弃了天然的 4 x 32 group层次，value/index状态也更大，优先级低于候选 B/C。

**踩坑与必要前提**

- 分层优化的关键不是“把 reduction 从 256 改成 32”，而是只对选中的 4 个 group做 local top-8。若先对全部 8 个 group计算 local top-8，再在 64 candidates上 merge，总比较量可能超过当前实现。
- 必须在片上完成 `4 x 32` compact。若重新从 GDRAM gather selected groups，算法减少的比较很可能被数据搬运抵消。
- 4-way merge 需要动态 cursor读取下一项；算法成本很低，但实际 lowering 可能产生昂贵 gather。该问题属于实现约束，不改变本 entry 的算法结论。
- compare-swap 必须同时交换 value 和 global expert ID，不能使用只返回 value 的排序而在事后猜测 ID。
- NaN 和 tie 行为必须先固定为明确的 total order；不能通过给 logit 添加 epsilon 破坏数值语义。
- TMO 的 `dim=(4,12,1)` 只能作为执行形态线索，不能证明其使用了上述分层算法或四核协作。

**结果**

本轮没有 kernel 实现和实测性能结果。纯算法分析得到以下优先级：

| 方案 | 理论比较量 | 串行深度 | 相对当前 | 判断 |
|---|---:|---:|---:|---|
| 当前 8 x argmax-256 | 2040 | 64 | baseline | 已实测 20.1264 us |
| A：local argmax + flat-32 merge | 1116（复用 top-1） | 75 | 工作量 -45.29%，深度 +17.19% | 不优先 |
| B：local argmax + 4-way merge | 892 | 51 | 工作量 -56.27%，深度 -20.31% | 首个实现候选 |
| C：bitonic sort-32 + 4-way merge | <=984 | 31 | 工作量 <=-51.76%，深度 -51.56% | 高收益、高风险 |

表中比较量和深度是算法模型，不等价于 MLU 指令数或 device time；不能据此直接宣称加速比例。

**与 upbound 的差距**

由于未实现，当前差距保持不变：`20.1264 / 9.9656 = 2.0196x`，绝对差 **10.1608 us**。本轮只证明存在同时减少比较量和依赖深度的精确算法结构，尚未证明 Triton backend 能高效表达。

**结论与下一步**

若进入实现阶段，应先验证候选 B：它不要求完整 sorting network，算法风险最低，并且 4-way merge 将 global selection 从 8 次 32-lane reduction 降为 8 次 4-way head比较。成功标准是保持全量正确性的同时进入 `<=18 us`。

候选 B 有收益后，再用 32-lane partial sorting network 替换 local repeated argmax，目标进入 `<=15 us`。如果 `4 x 32` compact 或 cursor merge 无法片上高效 lowering，则应停止堆叠完整 kernel，转而分别 microbenchmark compact、local list生成和 4-way merge。

## 5. 当前瓶颈判断

### 5.1 Expert top-8 的串行 reduction

当前 kernel 仍执行 8 次串行 256-lane indexed argmax。选中 group 后虽然只有 128 个有效 expert，但 MLU Triton 中已测试的 compact 方法会引入 gather、额外 GDRAM load 或昂贵 layout copy。

这是下一阶段最明确的算法瓶颈。

### 5.2 单 core/program 的执行模型

当前 launch 为 `dim=(48,1,1)`，每个 program独立处理 token。TMO 为 `dim=(4,12,1)`，看起来由每 cluster 4 个 core协同处理工作。TMO 很可能使用 BANG vector primitive、cluster 内协作和专用 top-k网络，这是 Triton 与 TMO 仍差约 2x 的主要候选原因。

### 5.3 Compiler lowering 约束

已经观察到：

- 默认 indexed argmax 能 lowering 到专用 NRAM argmax。
- fast tie-break 会退化为通用 `linalg.reduce`。
- 动态 compact load 会变成通用 gather。
- scalar收集和循环内 store/reload 代价很高。

因此后续优化不能只看 Triton 源码操作数，必须同时检查 `.linalgopt`、`.mlisa`、GPR/NRAM和 profiler duration。

### 5.4 Host launcher 仍占约 7 us

最终专用入口 wall 约 27.25 us、device约 20.13 us，两者差约 7.1 us。该差值包含 Python wrapper、校验、launcher 和同步摊销。三参数 entry 已明显改善，但若目标是 eager 端到端 latency，仍可提供内部 unsafe/prevalidated入口继续压缩 host路径。

## 6. 后续优化方向

按优先级排列：

### P0 - 建立分阶段 microbenchmark，量化真正热点

分别构造只包含以下阶段的临时 kernel，通过 duration差分定位成本：

1. load + group max
2. load + group rank
3. 增加 1/2/4/8 次 expert argmax
4. 增加 sparse normalization
5. 增加 masked output scatter

目标是得到“每次 256-lane indexed argmax”的边际成本，避免继续凭源码直觉优化。

### P0 - Expert top-8 层次化方案

候选方案：

- 在 `[8,32]` 上并行做每组 local top-k，再把选中 group的候选合并为 32/64 lanes做 global top-8。
- 重点不是减少比较总数，而是把 256-lane串行 reduction换成后端更擅长的 32/64-lane NRAM argmax。
- 必须保留 value/index pair，并确认 lowering 仍使用专用 argmax，而不是通用 reduce。

该方案是否有效需要 microbenchmark；从算法轮数看未必更少，但可能更符合 MLU vector primitive。

### P0 - NRAM 内 compact/layout primitive

重新研究 MLU Triton backend 是否有可控的 permute、gather、shared/NRAM layout或 extern libdevice primitive，使 256 load 后能在 NRAM 内把 4 个动态 group压成 128 candidates。

成功条件：

- 不再次读取 GDRAM。
- 不生成 `linalg_ext.gather`。
- 不生成大量 memref copy/interleave。
- 128-lane top-8 总时间低于当前 256-lane路径至少 2 us。

### P1 - Selection network / value-index pair sort

`tl.sort` 只返回 value，没有直接 argsort。可探索：

- 手写 compare-swap network，同时维护 float value 和 int ID。
- 对 32/64 lanes而非 256 lanes使用 selection network。
- 只生成 top-8 的 partial network，不做 full sort。

风险是 IR/MLISA爆炸和编译时间过长。必须从 32-lane microbenchmark开始，不能直接展开 256-lane网络。

### P1 - 多 core/cluster 协作

研究 MLU Triton 是否能表达与 TMO 类似的 `4 cores x 12 clusters` 协作：

- 2D grid、shared SRAM/cluster scratch、barrier或原子同步能力。
- 让 4 个 core分别处理 group/expert分片，再合并 local top-k。
- 若必须拆成两个 kernels，要把额外 launch和中间 GDRAM通信计入总成本。

这是最可能接近 TMO 10 us 的结构性方向，但实现和后端能力风险最高。

### P1 - 继续缩短固定 shape launcher

- 提供仅供内部 benchmark/上层已校验调用的 `unsafe_out` 入口，跳过 Python shape/dtype/device检查。
- 为其他高频 token 数生成独立三参数 kernel，而不是给通用 kernel增加 constexpr控制参数。
- 若上层可使用 compile/capture，比较专用 entry 在图内的实际 host成本。

该方向主要改善 wall time，不会缩小 20.13 us 与 9.97 us 的 device gap。

### P2 - 编译选项与资源调优

- 监控 GPR/NRAM变化，探索 backend的 shared promotion、FP fusion、on-chip分析选项。
- 只有在算法结构稳定后再做；此前 `num_warps/num_stages` 已证明收益有限。

## 7. 下一阶段目标

建议按以下里程碑评价后续结果：

| Milestone | Device time | 相对当前 | 相对 TMO | 评价 |
|---|---:|---:|---:|---|
| Current | 20.13 us | 1.00x | 2.02x | 当前基线 |
| M1 | <=18 us | >=10.6% lower | <=1.81x | 证明 expert top-k新结构有效 |
| M2 | <=15 us | >=25.5% lower | <=1.51x | Triton 有较强工程竞争力 |
| M3 | <=12 us | >=40.4% lower | <=1.20x | 接近 TMO/BANGC |
| Upbound | 9.97 us | 50.5% lower | 1.00x | 当前实测工程上界 |

任何新方案若不能在同一 trace 中稳定改善至少约 0.5 us，不应进入主实现；小于该幅度的结果需要增加迭代数、交错顺序并重复采集确认。

## 8. 当前文件与复现命令

### 文件

- 原始 PyTorch 实现：[base.py](base.py)
- 首版 Triton：[triton_grouped_topk.py](triton_grouped_topk.py)
- 当前优化 Triton：[triton_grouped_topk_optimized.py](triton_grouped_topk_optimized.py)
- 当前 benchmark：[benchmark_triton_grouped_topk_optimized.py](benchmark_triton_grouped_topk_optimized.py)
- 当前 profiler：[profile_triton_grouped_topk_optimized.py](profile_triton_grouped_topk_optimized.py)
- 最终 Triton trace：[final trace](log/triton_grouped_topk_group_rank_fixed_T83_preallocated_50iter.pt.trace.json)
- TMO upbound trace：[TMO trace](log/tmo_moe_softmax_topk_T83_preallocated_50iter.pt.trace.json)

### 命令

```bash
/projs/framework/lipenghui/venv/pytorch_main/bin/python \
  benchmark_triton_grouped_topk_optimized.py \
  --tokens 83 --warmup 20 --iterations 200 --repeats 5

/projs/framework/lipenghui/venv/pytorch_main/bin/python \
  profile_triton_grouped_topk_optimized.py
```

按 kernel name汇总 trace：

```bash
jq -r '
  .traceEvents[]
  | select(.cat == "kernel")
  | [.name, .dur, (.args.extra.dimx // "")]
  | @tsv
' log/triton_grouped_topk_group_rank_fixed_T83_preallocated_50iter.pt.trace.json
```

## 9. 后续 Entry 模板

后续每次实验追加一个 entry，不覆盖失败记录：

```markdown
### Entry NNN - 标题（成功 / 失败 / 部分成功）

**状态**

实验前的 best-known baseline 和瓶颈判断。

**假设与优化手段**

为什么可能更快，具体改了什么。

**踩坑**

编译、lowering、正确性、资源、测量口径中的问题。

**结果**

同 trace device time、交错 wall time、GPR/NRAM、正确性和产物链接。

**与 upbound 的差距**

latency ratio、absolute gap、throughput fraction。

**结论与下一步**

保留/淘汰，以及下一个可证伪的优化假设。
```

## 10. Checkpoint

记录生成时：2026-08-03。

- `base.py` 未修改，SHA256：`3166785ba472cca822335b9857a098ca12d4b773b48beff5773b5f64309d5db7`
- 当前 optimized Triton SHA256：`5c898c52575662f25ff0798b400f658a6e06442fabb5037a0e0f0f0f4433e7d1`
- 最终 trace SHA256：`7a0a41c2784425542ae773b989c329646a37ba5021ca48e35364e07f331e0c0f`

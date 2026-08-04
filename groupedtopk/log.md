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
| Entry 010 T=83 specialized Triton | 1 | 20.1264 us | 8.99% lower | 2.02x |
| 当前 compact-128 Triton | 1 | **19.0008 us** | 5.59% lower | **1.91x** |
| TMO/BANGC reference | 1 | **9.9656 us** | - | 1.00x |

当前 Triton 相对 compiled PyTorch 图的 device speedup 为 **5.92x**。从 compiled PyTorch 的 112.48 us 到 TMO 的 9.9656 us 这段可优化区间看，当前 Triton 已消除约 **91.19%** 的延迟差距。

但从当前 Triton 自身看，距离 TMO 仍有 **9.0352 us** 绝对差距：

- 当前 Triton latency / TMO latency：**1.9066x**
- 当前 Triton 等效吞吐 / TMO 吞吐：**52.45%**
- 若以 TMO 为目标，当前 latency 仍需再下降 **47.55%**

最终交错 wall-time 长测：

| 实现 | Wall time/call | 结果 |
|---|---:|---:|
| 首版 Triton preallocated launcher | 31.181 us | baseline |
| Entry 010 T=83 三参数 launcher | 27.253 us | 下降 12.60% |
| Entry 018 同轮 Entry 010 launcher | 28.249 us | same-run baseline |
| 当前 compact-128 launcher | **26.520 us** | **同轮下降 6.12%** |

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

### Entry 010 - T=83 三参数专用 kernel/launcher（阶段最佳）

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

### Entry 011 - 两级 winner tree（失败）

**状态**

Entry 010 在选出 group top-4 后，仍对 256 lanes执行 8 次串行 indexed argmax。同 trace baseline为 **20.0288 us**。

**优化手段**

实现两级 winner tree。每轮先在 8 个 group内并行执行 32-lane argmax，再对 8 个 group winner执行一次 8-lane argmax；输出 global winner后，只更新对应 group的 candidate。目标是用小宽度 reduction替代 256-lane reduction。

**踩坑**

源码层面的分层没有被 backend融合成单个层次化 primitive。每轮实际生成一个 32-lane argmax和一个 8-lane argmax，8 轮共形成 16 个 argmax及其 value/index状态更新。

**结果**

- device：20.0288 us -> **45.0560 us**，退化 **124.96%**。
- MLISA从约 937 行增加到 **3126 行**，GPR从约 674增加到 **2243**，NRAM约 **19.5 KB**。
- 随机、递增、递减、重复值和全相等输入正确性通过。
- trace：[hierarchical candidates trace](log/triton_grouped_topk_hierarchical_candidates_T83_preallocated_50iter.pt.trace.json)

**与 upbound 的差距**

- 相对 TMO：`45.0560 / 9.9656 = 4.5212x`。
- 绝对差距：**35.0904 us**；要达到 TMO仍需下降 **77.88%**。

**结论与下一步**

失败。MLU Triton对多个小 argmax的 lowering成本高于单个宽 argmax，不能只按算法比较量判断收益。后续不再使用重复的二维 winner tree。

---

### Entry 012 - sort-32 + sort-64（失败）

**状态**

Entry 011证明 repeated hierarchical argmax不能有效 lowering。本 entry尝试用固定 compare-swap network减少串行 top-k轮数，同 trace baseline仍为 **20.0288 us**。

**优化手段**

对每个 group执行 bitonic sort-32并保留 local top-8，再将 8组共 64 个 candidates执行 bitonic sort-64。使用 64-bit value/ID key同步保持 logit排序、global expert ID和稳定 tie顺序。

**踩坑**

完整 sorting network会在编译期展开全部 compare-swap stage；64-bit key又增加了 pair状态和指令宽度。虽然算法依赖深度下降，生成代码规模和寄存器压力远超 indexed argmax版本。

**结果**

- device：20.0288 us -> **170.6424 us**，退化 **751.99%**。
- 编译资源：**5344 GPR / 46.1 KB NRAM / 15445 行 MLISA**。
- 随机和 edge case的 ID、weight正确性通过。
- trace：[hierarchical candidates trace](log/triton_grouped_topk_hierarchical_candidates_T83_preallocated_50iter.pt.trace.json)

**与 upbound 的差距**

- 相对 TMO：`170.6424 / 9.9656 = 17.1231x`。
- 绝对差距：**160.6768 us**；要达到 TMO仍需下降 **94.16%**。

**结论与下一步**

失败。不能在完整 grouped top-k kernel中直接展开 full bitonic sort。后续若继续 selection network，只能从独立 32/64-lane partial top-8 microbenchmark开始，并严格控制 comparator和 value/index pair规模。

---

### Entry 013 - 通用 `tl.gather` 片上 compact-128（失败）

**状态**

前两条分层路线都因 lowering膨胀失败。本 entry回到 indexed argmax，只尝试把已选中的 4 个 32-expert group压缩为 128 candidates。同 trace baseline为 **20.0288 us**。

**优化手段**

复用第一次 load后的 logits tensor，根据动态 selected group ID构造 128 个 source offset，并使用通用 `tl.gather`生成 compact candidate tensor；随后执行 8 次 128-lane indexed argmax，不重新读取 GDRAM。

**踩坑**

虽然 reduction宽度从 256降到 128，但动态 `tl.gather`没有 lower成适合连续 group window的廉价搬运，compact控制和数据重排成本超过 reduction节省。

**结果**

- device：20.0288 us -> **21.9048 us**，退化 **9.37%**。
- compact语义和所有正确性用例通过。
- trace：[hierarchical candidates trace](log/triton_grouped_topk_hierarchical_candidates_T83_preallocated_50iter.pt.trace.json)

**与 upbound 的差距**

- 相对 TMO：`21.9048 / 9.9656 = 2.1980x`。
- 绝对差距：**11.9392 us**；要达到 TMO仍需下降 **54.50%**。

**结论与下一步**

失败。片上 compact方向仍有理论收益，但通用 gather表达不合适。下一步利用每个 group都是连续 32 elements这一结构，改用 MLU backend的连续-window gather。

---

### Entry 014 - 连续-window gather + prefix rank（失败）

**状态**

Entry 013的主要开销来自通用 gather。本 entry保持 compact-128算法不变，只替换数据搬运 primitive和 selected group收集逻辑。同 trace baseline为 **20.1192 us**。

**优化手段**

使用 `triton.language.extra.mlu.gather`从已加载的 NRAM tensor搬运 4 个连续 32-element window；用 8x8 prefix比较为 selected group计算 compact slot。

**踩坑**

连续-window gather确实减少数据重排开销，但 8x8 prefix rank引入额外比较、mask和中间 tensor。小规模控制逻辑在源码上很短，不代表会 lower成廉价标量指令。

**结果**

- device：20.1192 us -> **21.2000 us**，退化 **5.37%**。
- 相对通用 gather的 21.9048 us改善 **3.22%**，但仍未超过 baseline。
- trace：[window-gather trace](log/triton_grouped_topk_hierarchical_window_gather_T83_preallocated_50iter.pt.trace.json)

**与 upbound 的差距**

- 相对 TMO：`21.2000 / 9.9656 = 2.1273x`。
- 绝对差距：**11.2344 us**；要达到 TMO仍需下降 **52.99%**。

**结论与下一步**

失败，但确认 backend连续-window gather优于通用 gather。下一步只优化 selected group ID的 8-to-4收集过程。

---

### Entry 015 - sort-4 收集 selected group（失败）

**状态**

Entry 014已把瓶颈缩小到 selected group ID收集。本 entry尝试固定小排序网络，同 trace baseline为 **20.2304 us**。

**优化手段**

根据 group rank得到 4 个 selected group后，使用手写 sort-4 compare-swap将 group ID整理为升序，再驱动连续-window gather，以保持 global expert ID tie顺序。

**踩坑**

sort-4需要从动态 lanes抽取 value/ID并展开 compare-swap。控制 tensor和 lane搬运成本明显大于四个标量比较的源码直觉。

**结果**

- device：20.2304 us -> **24.9296 us**，退化 **23.23%**。
- trace：[sort-4 trace](log/triton_grouped_topk_compact128_sort4_T83_preallocated_50iter.pt.trace.json)

**与 upbound 的差距**

- 相对 TMO：`24.9296 / 9.9656 = 2.5016x`。
- 绝对差距：**14.9640 us**；要达到 TMO仍需下降 **60.03%**。

**结论与下一步**

失败。selected group已经天然按 group lane分布，不应抽取后再次排序。下一步利用 lane顺序做 prefix/cumsum压缩。

---

### Entry 016 - cumsum 收集 selected group（失败）

**状态**

Entry 015的显式排序代价过高。本 entry用 selected mask的 prefix sum直接计算 compact slot，同 trace baseline为 **19.9832 us**。

**优化手段**

在 8 个 group lanes上对 selected mask执行 cumsum，得到每个 selected group的 0..3 compact position，再生成升序 selected group ID。

**踩坑**

8-lane cumsum仍会生成 reduction/prefix相关指令。它比 8x8 prefix和 sort-4更好，但控制路径成本仍超过 128-lane argmax节省。

**结果**

- device：19.9832 us -> **20.9864 us**，退化 **5.02%**。
- trace：[cumsum trace](log/triton_grouped_topk_compact128_cumsum_T83_preallocated_50iter.pt.trace.json)

**与 upbound 的差距**

- 相对 TMO：`20.9864 / 9.9656 = 2.1059x`。
- 绝对差距：**11.0208 us**；要达到 TMO仍需下降 **52.51%**。

**结论与下一步**

失败。下一步使用 backend已有的 `tl.masked_select`直接压缩 selected group ID，避免手写 prefix逻辑。

---

### Entry 017 - `masked_select` + 二次 gather（失败）

**状态**

Entry 016表明手写 prefix仍然太贵。本 entry改用 `tl.masked_select`，同 trace baseline为 **20.0384 us**。

**优化手段**

用 `tl.masked_select(group_offsets, selected_groups)`按原始 group ID顺序压缩 selected IDs，再通过一次小 gather取得前 4 项，最后执行连续-window gather和 8次 128-lane argmax。

**踩坑**

`masked_select`本身有效，但为了从其 8-slot结果抽取前 4 项而增加的第二次动态 gather抵消了收益。

**结果**

- device：20.0384 us -> **20.4568 us**，退化 **2.09%**。
- 这是首个接近 baseline的 compact-128控制路径。
- trace：[masked-select trace](log/triton_grouped_topk_compact128_masked_select_T83_preallocated_50iter.pt.trace.json)

**与 upbound 的差距**

- 相对 TMO：`20.4568 / 9.9656 = 2.0527x`。
- 绝对差距：**10.4912 us**；要达到 TMO仍需下降 **51.28%**。

**结论与下一步**

失败，但已定位最后的额外成本。下一步必须保持 `masked_select`，同时消除第二次动态 gather。

---

### Entry 018 - `masked_select` + fixed reshape compact-128（成功）

**状态**

Entry 017只比 baseline慢 0.4184 us，剩余明显冗余是 selected ID上的第二次 gather。同 trace baseline为 **20.0432 us**。

**优化手段**

保留 `tl.masked_select`按原始 group ID顺序压缩 group ID，将 8-slot结果 reshape为 `[2,4]`，通过固定 lane选择取得第一行，消除第二次动态 gather。随后用 MLU连续-window gather从第一次 load后的 NRAM tensor一次搬运 4 个 32-element window，再执行 8次 128-lane indexed argmax。整个过程不重新读取 GDRAM。

**踩坑**

固定选择不能破坏 ascending group ID顺序，否则 equal-logit时会改变 global expert ID tie-break。最终实现额外验证了 group score顺序不同于 group ID顺序的 cross-group tie输入。

**结果**

| Kernel | Count | Average | Min | Max |
|---|---:|---:|---:|---:|
| Entry 010 `_grouped_topk_group_rank_t83_kernel` | 50 | 20.0432 us | 19.64 us | 20.60 us |
| Entry 018 `_grouped_topk_compact128_t83_kernel` | 50 | **19.0008 us** | 18.72 us | 19.60 us |

- device latency下降 **5.20%**，speedup **1.0549x**。
- 第二份独立 trace：20.1480 us -> **19.0192 us**，下降 **5.60%**。
- compact kernel约使用 **680 GPR / 8.8 KB NRAM**，MLISA约 973 行。
- 交错 wall长测：28.249 us -> **26.520 us**，中位数下降 **6.12%**。
- 最终 trace：[compact-128 final trace](log/triton_grouped_topk_compact128_masked_select_final_T83_preallocated_50iter.pt.trace.json)
- 独立复核：[compact-128 repeat trace](log/triton_grouped_topk_compact128_masked_select_final_repeat_T83_preallocated_50iter.pt.trace.json)
- benchmark：[benchmark_triton_grouped_topk_hierarchical.py](benchmark_triton_grouped_topk_hierarchical.py)
- profiler：[profile_triton_grouped_topk_hierarchical.py](profile_triton_grouped_topk_hierarchical.py)

**正确性**

- 3 个随机 seed的 IDs完全一致，最大权重误差不超过 `5.96e-8`。
- 递增、递减、重复值和全相等输入与 Entry 010完全一致。
- group score顺序不同于 group ID顺序且 expert value tie的构造输入完全一致。

**与 upbound 的差距**

- 相对 TMO：`19.0008 / 9.9656 = 1.9066x`。
- 等效吞吐为 TMO的 **52.45%**。
- 绝对差距：**9.0352 us**；达到 TMO仍需再下降 **47.55%**。

**结论与下一步**

成功。MLU Triton可以用连续-window gather在 NRAM内高效 compact动态选中的连续 group。Entry 005的结论应收窄为“通用 gather或重新从 GDRAM load失败”。当前最佳更新为 **19.0008 us**；下一步在 128 candidates上减少 8次串行 argmax。

---

### Entry 019 - pipeline/backend meta参数 sweep（失败）

**状态**

Entry 018已得到当前最佳 19.0008 us。本 entry检查是否能通过 backend调度参数继续降低 device time。

**优化手段**

分别测试 `bottleneck=none/io/mv/simd`、`force_bottleneck`和 `num_stages=2`，保持 grid=48和算法不变。

**踩坑**

这些参数没有改变 8次 128-lane argmax的串行依赖，也没有形成可重叠的多阶段 load/compute流水。

**结果**

各配置相对默认值的差异小于 **0.1 us**，没有稳定超过 Entry 018；最终保留默认 `num_stages=1`和 grid=48。实验 trace保存在 `log/triton_grouped_topk_compact128_{bottleneck,pipeline}_*.pt.trace.json`，不作为当前结果 trace。

**与 upbound 的差距**

无稳定收益，当前最佳和 TMO差距保持为 **1.9066x / 9.0352 us**。

**结论与下一步**

失败。当前瓶颈不是可由 meta参数解决的调度问题；继续优化应转向 128-candidate partial selection算法。

---

### Entry 020 - U1 row-max batch tile（部分成功，未升级为完整 kernel）

**现状**

Entry 018的完整 kernel仍以单 token、单 core program执行。先把只负责 256 个 group score 的 row-max阶段抽出，验证一个 U1 program是否可以批量处理多个 token，并量化批处理本身的收益。

**优化手段**

新增 `triton_grouped_topk_u1.py`。单 token版本使用 `grid=48`、`num_warps=1`；批处理版本使用 `[8,256]` tile、`grid=12`，每个 program处理 8 个 token。分别测试 `num_warps=1`、`num_warps=4` 和 shared-memory meta。

**踩坑**

MLU `fast_libentry` 会按函数对象缓存第一次编译的 meta配置；复用同一个 runner测试 `num_warps=4`会错误地继续执行 `num_warps=1` 的 cnbin。测试脚本改为每种 meta使用独立 runner，并用 MLISA确认实际编译参数。

**结果**

| Kernel | Device time |
|---|---:|
| 单 token row-max, w1 | 21.4628 us |
| U1 batch-8 row-max, w1 | 17.5160 us |
| U1 batch-8 row-max, w4 | 17.8104 us |
| U1 batch-8 row-max, w4 + shared meta | 17.7724 us |

批处理 row-max 相对单 token speedup 为 **1.2254x**（18.38%）。`w4` 虽然生成 SRAM/N策略不同的 MLISA，但没有超过 `w1`；shared meta也没有改变实际 `promote_shared`。

**与 upbound 的差距**

这是 row-max 子阶段，不与完整 TMO device time直接比较。它证明 batch tile能摊薄 launch和部分索引开销，但完整 top-8仍需保留串行 reduction。

**结论与下一步**

部分成功，作为独立 microbenchmark保留，不替换 Entry 018的生产路径。下一步验证同样的 batch U1映射是否能摊薄完整 top-8；若 argmax展开成本随 batch增长，则该路线失败。

---

### Entry 021 - 完整 U1 batch-8 grouped top-k（失败）

**现状**

基于 Entry 020的 batch-8映射实现完整 grouped top-k，输入为 `[8,256]`，每个 program处理 8 个 token；输出保留 dense `[8,8]` top-k结果，和 Entry 018相同。

**优化手段**

新增 `triton_grouped_topk_batched_u1.py`。将 group score、4组 compact、128 candidate load、8轮 top-1 reduction和 dense output全部向量化到 batch维，使用 `grid=12`。分别测试 `num_warps=1` 和 `num_warps=4`。

**踩坑**

batch化会把 8 次串行 argmax展开成 64 个独立的 `argmax.nan`，没有减少关键路径。`w1` 版本 MLISA约 `168960 B NRAM / 2672 GPR`；`w4`版本约 `27904 B NRAM / 131200 B SRAM`，资源下降并没有抵消 reduction和 dense output的额外指令。

**结果**

| Kernel | Device time |
|---|---:|
| Entry 018 compact128 | 18.7472 us |
| U1 batch-8, w1 | 44.5560 us |
| U1 batch-8, w4 | 28.5448 us |

相对 baseline（Entry 018）speedup：`0.4206x`（w1）和 `0.6564x`（w4）。三组输入输出均通过 reference correctness，问题是 device time而非算法正确性。

**与 upbound 的差距**

Entry 018相对 TMO约 `1.9066x`；batch-8 w4退化到约 `2.864x` TMO，距离上界进一步扩大。

**结论与下一步**

失败。U1 batch化不能直接包住完整 top-k；后续转向减少动态 group compact或测试明确的 SRAM promotion，而不是继续扩大 batch。

---

### Entry 022 - 完整 U1 batch-2 / batch-4 grouped top-k（失败）

**现状**

为排除 batch-8 的展开规模问题，新增 `triton_grouped_topk_batched_sweep.py`，测试更小的 batch tile：B=2 使用 `grid=48`，B=4 使用 `grid=24`，均保持 `num_warps=1`。

**优化手段**

对 batch 维采用编译期 `BLOCK_ROWS`，其余 grouped top-k流程与 Entry 021一致，比较不同 batch规模的 reduction、NRAM和 launch摊销。

**踩坑**

即使 B=2，每个 program仍需执行 16 轮独立 argmax；B=4则为32轮。批处理只减少 program数量，没有减少每个 token的 top-k关键路径。

**结果**

| Kernel | Device time |
|---|---:|
| Entry 018 compact128 | 18.7472 us |
| U1 batch-2 | 23.2876 us |
| U1 batch-4 | 30.4204 us |

相对 baseline speedup：`0.8050x`（B=2）和 `0.6163x`（B=4）。IDs和权重均与 Entry 018一致。

**与 upbound 的差距**

B=2约为 TMO的 `2.266x`，B=4约为 `2.959x`；均未缩小 Entry 018相对 TMO的 `1.9066x`差距。

**结论与下一步**

失败。完整 kernel不适合用简单 batch U1映射继续摊薄；下一轮应只改 compact/索引路径，保持单 token的 top-k reduction结构。

---

### Entry 023 - fixed prefix compaction 与 shared promotion（失败）

**现状**

Entry 018使用 `tl.masked_select` 后再固定 reshape，当前完整 kernel在本轮同一 trace中的 device time为 **14.6524 us**。本 entry分别验证固定宽度 prefix compaction和 MLU compiler的 `force_use_shared_memory`。

**优化手段**

新增 `triton_grouped_topk_compact_variants.py`。prefix版本用 `[8,4]` one-hot比较和归约计算 ascending group-ID的4个 compact slot，保持 tie-break语义；shared版本复用 Entry 018 kernel，只传 `force_use_shared_memory=True`，不改变算法。

**踩坑**

`force_use_shared_memory`只有在 `num_warps=1` 且 backend允许 promotion时才会生效；MLISA JSON确认 shared版本的 `promote_shared=true`。prefix版本虽然没有动态 `masked_select`，但固定 one-hot会增加一组 tile和寄存器压力。

**结果**

| Kernel | Device time | 相对 baseline speedup |
|---|---:|---:|
| Entry 018 compact128 | 14.6524 | 1.0000x |
| fixed prefix compaction | 17.1568 | 0.8541x |
| compact128 + shared promotion | 14.6720 | 0.9987x |

prefix相对 baseline退化 **17.09%**；shared promotion仅差 **0.0196 us**，没有稳定收益。prefix MLISA约 **832 GPR / 9344 B NRAM**，baseline约 **680 GPR / 9024 B NRAM**。

**与 upbound 的差距**

TMO参考 device time为 **9.9656 us**。baseline为 `1.4704x` TMO，prefix为 `1.7216x`，shared为 `1.4727x`；两条路线都没有缩小差距。

**结论与下一步**

失败。当前 `masked_select + fixed reshape` 已经比手写 one-hot prefix 更适合 MLU Triton lowering，shared promotion也不是主要瓶颈。下一轮保持 compact路径不变，只尝试减少 top-k结果重排和 masked scatter指令。

---

### Entry 024 - dense top-k result accumulation（失败）

**现状**

Entry 018在 128 candidates上执行8轮 indexed argmax，每轮写入 128-lane `selected_rank`，最后用 rank做一次 masked scatter。这个结果重排可能是剩余的 Triton控制开销之一。

**优化手段**

新增 `triton_grouped_topk_direct_topk.py`。保留 Entry 018 的 group compact和连续-window gather，每轮直接把 `best_value` 和通过 mask归约得到的 `best_id` 放入固定的8-lane `top_values/top_ids`，循环结束后直接 dense store，不再维护 128-lane `selected_rank`。

**踩坑**

MLU Triton没有直接暴露动态 scalar lane读取；为了保持 candidate ID和 value同步，只能对 `candidate_offsets == best_position`执行 128-lane mask归约。这个额外 reduction抵消了省掉的 selected-rank状态。

**结果**

| Kernel | Device time | 相对 baseline speedup |
|---|---:|---:|
| Entry 018 compact128 | 14.6492 | 1.0000x |
| direct dense top-k | 15.9216 | 0.9202x |

IDs完全一致，最大权重误差为 `5.96e-8`。direct版本相对 baseline退化 **8.68%**；MLISA约 **1870 GPR / 2944 B NRAM / 1924 行**，明显高于 compact128的约 **680 GPR / 9024 B NRAM**。

**与 upbound 的差距**

TMO参考 device time为 **9.9656 us**。baseline为 `1.4703x` TMO，direct dense为 `1.5977x`，没有缩小差距。

**结论与下一步**

失败。`best_id` 的动态提取比原有 128-lane selected-rank + 一次 masked scatter 更昂贵。后续不再在 Triton层手写 scalar dynamic gather；若要继续降低8轮 reduction，需要专门的 partial-select primitive或更底层的 device pipeline语义。

## 5. 当前瓶颈判断

### 5.1 Expert top-8 的串行 reduction

当前 kernel 已将选中 group片上 compact为 128 candidates，但仍执行 8 次串行 128-lane indexed argmax。相对 Entry 010 已减少一半 reduction宽度，下一阶段需要减少 argmax轮数或进一步层次化 selection。

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

最终 compact入口 wall约 26.52 us、device约 19.00 us，两者差约 7.5 us。该差值包含 Python wrapper、校验、launcher和同步摊销。若目标是 eager端到端 latency，仍可提供内部 unsafe/prevalidated入口继续压缩 host路径。

## 6. 后续优化方向

按优先级排列：

### P0 - 建立分阶段 microbenchmark，量化真正热点

分别构造只包含以下阶段的临时 kernel，通过 duration差分定位成本：

1. load + group max
2. load + group rank
3. 增加 1/2/4/8 次 expert argmax
4. 增加 sparse normalization
5. 增加 masked output scatter

目标是得到“每次 128-lane indexed argmax”的边际成本，避免继续凭源码直觉优化。

### P0 - 128-candidate partial selection network

在已经 compact的 128 candidates上，从 32/64-lane partial compare-swap network开始，逐步替换 8 次串行 argmax。完整 bitonic sort-32/64已经证明不可行，下一轮必须只生成 top-8需要的 comparator，并分别 microbenchmark value和 value/index pair。

### P0 - 继续压缩 compact控制路径

当前 `masked_select + fixed reshape + window gather` 已实现片上 compact。后续只比较更低成本的 selected-group ID编码或直接复用 group-rank结果；成功条件是保持 tie语义并在同 trace中改善至少 0.5 us。

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

该方向主要改善 wall time，不会缩小 19.00 us 与 9.97 us 的 device gap。

### P2 - 编译选项与资源调优

- 监控 GPR/NRAM变化，探索 backend的 shared promotion、FP fusion、on-chip分析选项。
- 只有在算法结构稳定后再做；此前 `num_warps/num_stages` 已证明收益有限。

## 7. 下一阶段目标

建议按以下里程碑评价后续结果：

| Milestone | Device time | 相对当前 | 相对 TMO | 评价 |
|---|---:|---:|---:|---|
| Current | 19.00 us | 1.00x | 1.91x | 当前基线 |
| M1 | <=18 us | >=5.3% lower | <=1.81x | 证明 expert top-k新结构有效 |
| M2 | <=15 us | >=21.1% lower | <=1.51x | Triton 有较强工程竞争力 |
| M3 | <=12 us | >=36.8% lower | <=1.20x | 接近 TMO/BANGC |
| Upbound | 9.97 us | 47.5% lower | 1.00x | 当前实测工程上界 |

任何新方案若不能在同一 trace 中稳定改善至少约 0.5 us，不应进入主实现；小于该幅度的结果需要增加迭代数、交错顺序并重复采集确认。

## 8. 当前文件与复现命令

### 文件

- 原始 PyTorch 实现：[base.py](base.py)
- 首版 Triton：[triton_grouped_topk.py](triton_grouped_topk.py)
- Entry 010 Triton：[triton_grouped_topk_optimized.py](triton_grouped_topk_optimized.py)
- 当前 compact-128 Triton：[triton_grouped_topk_hierarchical.py](triton_grouped_topk_hierarchical.py)
- 当前 benchmark：[benchmark_triton_grouped_topk_hierarchical.py](benchmark_triton_grouped_topk_hierarchical.py)
- 当前 profiler：[profile_triton_grouped_topk_hierarchical.py](profile_triton_grouped_topk_hierarchical.py)
- 最终 Triton trace：[compact-128 final trace](log/triton_grouped_topk_compact128_masked_select_final_T83_preallocated_50iter.pt.trace.json)
- TMO upbound trace：[TMO trace](log/tmo_moe_softmax_topk_T83_preallocated_50iter.pt.trace.json)

### 命令

```bash
/projs/framework/lipenghui/venv/pytorch_main/bin/python \
  benchmark_triton_grouped_topk_hierarchical.py \
  --warmup 20 --iterations 300 --repeats 7

/projs/framework/lipenghui/venv/pytorch_main/bin/python \
  profile_triton_grouped_topk_hierarchical.py
```

按 kernel name汇总 trace：

```bash
jq -r '
  .traceEvents[]
  | select(.cat == "kernel")
  | [.name, .dur, (.args.extra.dimx // "")]
  | @tsv
' log/triton_grouped_topk_compact128_masked_select_final_T83_preallocated_50iter.pt.trace.json
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
- 当前 compact-128 Triton SHA256：`fbeec4189086e2c2197247c2b7fc70c518cc0d3fa78174c0760d696e4719f4b9`
- 最终 trace SHA256：`ce0f61dcbf8e5cae7b33c50d35bd6dc490afe575464781964879ca6d835af92c`

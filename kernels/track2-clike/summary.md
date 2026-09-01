# Track2 C-like 算子优化总结

本文是赛道二 `sparse_attn`、`index_topk` 与 `sinkhorn_normalize` 的结果、源码、证据和工程经验总览。它保留当前可引用的 canonical 实现，并将结论对应到最小原始证据包；不把完整 campaign 的 mutable state、probe、日志或构建产物带回仓库。

## 口径与范围

- **数据来源**：本文所有性能和验证结论来自已验证的闭合归档 `kernelswift-track2-ascendc-round006-closed.tar.gz`，SHA-256 为 `eaf64ea5f5c926dcb5e5d1c4d0cc974f486a87b939c72a715ee05d1081d35fea`。本次整理没有重新执行 Ascend 硬件测量、编译或正确性测试。
- **性能口径**：正式数值使用 `auto_bench.py` 的完整算子 wall-time 中位数；`speedup = base_median_ms / candidate_median_ms`。局部 primitive、子 kernel 或仅 launcher 的时延不能代替完整算子结果。
- **正确性口径**：official nominal correctness 使用 seed `42`、`atol=1e-2`、`rtol=1e-2` 与 `equal_nan=True`。静态 binding、CMake build、`.so` load / launcher export、最小正确性、nominal correctness、语义 guardrail、screening、正式 timing、profiler / lowering 与 canonical promotion 是逐级不同的证据；低层 PASS 不能跳过高层门槛。
- **canonical 条件**：实现必须通过 nominal correctness、适用的语义 guardrail 与输出契约；只有满足性能采纳条件的候选才会替换当前 canonical。可加载、可编译或某个最小样例通过，均不是 promotion。
- **证据冲突处理**：归档中早期 `sparse_attn/README.md` 是 qualification-only 快照，关于“未 benchmark”的说法已经过时。sparse baseline 的权威状态以本 summary 链接的 baseline report、最终 failure report / verdict 与根目录结果表为准。
- **证据边界**：`evidence/` 中的文件是从闭合归档逐字节复制的 provenance snapshot。为保持证据原貌，其正文可能提及未保留的历史路径；这些内部引用不是当前紧凑树的依赖，也不保证可解析。

## 结果总览

| 算子 | 当前 canonical | 记录的完整算子结果 | 采纳状态 |
|---|---|---:|---|
| `sparse_attn` | [`baseline_adapter.py`](sparse_attn/ascendc/baseline_adapter.py) | base `12.773625 ms`；adapter baseline `12.772110 ms` | baseline accepted；没有已升级的原生 candidate |
| `index_topk` | eager [`candidate_007.py`](index_topk/ascendc/candidate_007.py) | `8.784070 → 8.298150 ms`，`1.059x` | correctness PASS，eager canonical |
| `sinkhorn_normalize` | [`candidate_001.py`](sinkhorn_normalize/ascendc/candidate_001.py) + [`sinkhorn_normalize.cpp`](sinkhorn_normalize/ascendc/sinkhorn_normalize.cpp) | `1.524825 → 0.484020 ms`，`3.1503450270657824x` | accepted native Ascend C implementation |

## 源码与证据索引

| 算子 | 参考 / canonical 源码 | 最小证据包 |
|---|---|---|
| `sparse_attn` | [`base.py`](sparse_attn/base.py) · [`baseline_adapter.py`](sparse_attn/ascendc/baseline_adapter.py) | [baseline report](sparse_attn/evidence/baseline_report.md) · [candidate failure](sparse_attn/evidence/candidate_failure.md) · [failure verdict](sparse_attn/evidence/candidate_failure_verdict.json) |
| `index_topk` | [`base.py`](index_topk/base.py) · eager [`candidate_007.py`](index_topk/ascendc/candidate_007.py) | [detailed evidence](index_topk/evidence.md) |
| `sinkhorn_normalize` | [`base.py`](sinkhorn_normalize/base.py) · [`candidate_001.py`](sinkhorn_normalize/ascendc/candidate_001.py) · [`CMakeLists.txt`](sinkhorn_normalize/ascendc/CMakeLists.txt) · [`sinkhorn_normalize.cpp`](sinkhorn_normalize/ascendc/sinkhorn_normalize.cpp) | [accepted report](sinkhorn_normalize/evidence/accepted_report.md) · [accepted verdict](sinkhorn_normalize/evidence/accepted_verdict.json) |

## 算子结论

### `sparse_attn`：语义 guardrail 是性能之前的硬门槛

当前 canonical [`baseline_adapter.py`](sparse_attn/ascendc/baseline_adapter.py) 是与原始 sparse attention 等价的 PyTorch adapter。固定 scope 为 `q [8,2600,64,128] bf16`、`kv [8,32,128] bf16`、`topk_idxs [8,2600,16] int32`、`attn_sink [64] fp32`，输出为 `[8,2600,64,128] bf16`。它处理 top-k indexed sparse KV：`-1` 必须完全无效，既不进入 value numerator，也不进入普通 softmax denominator；duplicate index 的每次出现都必须贡献；attention sink 只参与分母而不写入加权 value 输出。归档记录中，base 的中位数为 `12.773625 ms`，adapter baseline 为 `12.772110 ms`；测量使用 warmup `200`、repeat `500`。这是一条语义 baseline 对照，不宣传为优化 speedup。

baseline 的 official nominal accuracy 在 seed 42、`atol=rtol=1e-2` 下通过，同时通过 mixed-invalid `-1`、all-invalid exact-zero、duplicate indices、denominator-only nonzero sink、输入不变性和输出不 alias 等 guardrail。这些是该题计算边界的一部分，而非额外测试。baseline profiler 记录为每 call `34` 个 AI Core task、summed device duration `12964.3378 us`；主要热点是 Cast `4120.9486 us`、BatchMatMul `3054.0648 us` 与 Index `2341.8264 us`。CANN task duration 可以重叠，summed-device / wall 大于 1 不代表设备利用率超过 100%。

Round 006 原生 candidate 的 load/export、exact launcher resolution、20800-block launch path 和 all-invalid 最小检查均通过；后者覆盖 exact-zero output、shape、dtype、layout、non-alias 与输入不变性。但正式 nominal correctness 失败：最大绝对误差 `3.165039`，平均绝对误差 `0.1596805`，`0 passed, 1 failed`。fail-fast 后 mixed-invalid、duplicate-index、arbitrary nonzero sink、non-default/current-stream sentinel、synchronization sentinel、screening、timing、profiling 和 observed lowering 都未运行，也不存在 candidate median 或 device duration。最终 verdict 是 `classification=code-error`、`terminal_result=candidate-failed`、`rule_id=CODE.CORRECTNESS.FAIL`、high confidence；它不能升级或作为原生性能结果引用。

在 Round 006 之前，candidate 001 虽通过 correctness 与语义 guardrail，但短筛选约为 `4425–4426 ms/call`，相对 `12.772110 ms` baseline 极端回退，已 `screened-out`；candidate 002 因地址对齐失败，candidate 003 因 launcher symbol 缺失，candidate 004 因数值错误失败；candidate 005 因 CANN 9.0.0 普通 vector caller 没有可用的 scalar `expf` 路径而在设计阶段终止，未执行也没有 timing。这些是不同类型的终止边界，不能合并成“原生路径已成功”或“没有证据”。

**结论**：all-invalid 或库加载成功只能验证局部链路。像 sparse attention 这类包含 masking、重复 index、sink 分母语义和输出精度的算子，必须以 nominal full-operator correctness 作为进入性能阶段的前提。

### `index_topk`：固定状态预计算比不合适的原生切分更有效

当前 canonical [`candidate_007.py`](index_topk/ascendc/candidate_007.py) 是 eager PyTorch 实现，不是原生 Ascend C kernel。它在初始化阶段预计算 causal mask 和每个 token 的有效 candidate 数，删除 forward 内固定的 `arange`、比较、整除和 mask 构造；mask 采用 BF16（`0` 与 `-inf` 均可精确表示），与 BF16 scores 对齐以避免额外 cast 和读取。`scores.relu_().mul_(...)` 采用原地路径，`start_pos=0, offset=0` 使用精确快路径避免无效逐元素 offset 加法；核心索引选择仍保留 `torch.topk` 的 tie-selection 语义。

它的 paired 记录为 `8.801935 → 8.295765 ms`、`1.061x`，稳定复测为 `8.784070 → 8.298150 ms`、`1.059x`，accuracy PASS。对于固定 shape、固定压缩率和固定 causal 规则，搬移不随输入变化的状态可以减少完整调用路径上的开销，而不改变 top-k 选择语义。

原生 post-BMM candidate 006 通过 `.so` build/load、Phase A BF16 product 位级全等、Phase B final indices 全等与完整短 paired benchmark accuracy，但端到端测得 `8.788365 → 14.463780 ms`、`0.608x`，故被拒绝。其问题不是“原生代码不正确”，而是仍会写入并读回完整 `[8,2600,16,650]`（约 `2.16` 亿 BF16 element、约 `432 MB`）张量，head reduction 留在原生 kernel 外，同时引入 BF16/FP32 转换、数据搬运、16 个 head 的串行处理以及 event synchronization。candidate 008 也通过 accuracy，但为 `0.827x`，同样未采纳。自定义 masked-topk 路径仍存在 `681` 个 tie-selection index 差异，首个差异对应分数完全相等，能力结论保持 `Unknown`，不能代替现有 `torch.topk` 路径。

已记录的 profiler 热点为 BatchMatMul `2.2411 ms`、InplaceMul `1.9925 ms`、InplaceRelu `1.3306 ms`、ReduceSum `0.9622 ms` 和 TopK `0.6010 ms`；其中 Relu + Mul + 16-head ReduceSum 合计约 `4.2853 ms`。真正的原生机会必须将 ReLU、head-weight multiply 和 16-head reduction 融合后直接输出 `[8,2600,650]`，才可能避开大张量写回与额外 launch；但这会改变累加顺序、BF16 中间舍入与最终 top-k index，因此必须继续以 exact discrete index semantics 验证。

**结论**：优化决策应以完整 tensor 生命周期、归约位置、同步和转换成本为单位。把一个中间算子原生化并不自动减少端到端成本；先消除固定 eager 开销在该题中更可靠。

### `sinkhorn_normalize`：完整迭代链融合需要 ABI 与语义同时闭合

[`candidate_001.py`](sinkhorn_normalize/ascendc/candidate_001.py) 是被接受的原生实现入口。它与 [`CMakeLists.txt`](sinkhorn_normalize/ascendc/CMakeLists.txt) 及 [`sinkhorn_normalize.cpp`](sinkhorn_normalize/ascendc/sinkhorn_normalize.cpp) 共同构成 Python + CMake + Ascend C 包：Python 以 `SOC_VERSION=ascend910b4`、`CMAKE_BUILD_TYPE=Release`、`ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0` 和当前 Python interpreter 配置构建，加载 `build/lib/libsinkhorn_normalize.so`，解析 `aclrtlaunch_sinkhorn_normalize`，并以 `torch.npu.current_stream().npu_stream` 在调用方 current stream 上发射内核。CMake build 遇到“对象文件目录为空”但 host object 已生成的特定工具链情形时，adapter 有受限 host-object fallback；这是一条固定 CANN / Ascend910B4 的兼容路径，不是跨版本 capability claim。

该实现固定 `repeat=10`、`eps=1e-6`，要求 contiguous NPU float32 `[1,1024,4,4]` 输入；不应推广到任意矩阵大小、repeat 或 eps。原生 kernel 使用 32 个 block，每个 block 处理 32 个 `4×4` matrix（512 个 FP32 element），从 GM 拷入 UB，在一个 local state buffer 中完成全部迭代后一次性写回 GM。精确序列是 stable row softmax（先减 row max）、softmax 后加 eps、一次 column normalization、再九次 row-normalization + column-normalization pair；共 19 个 normalization denominator 使用 eps。

它把 59 个 eager kernel 融合为 1 个原生 kernel。三组 interleaved reference/candidate 样本分别为 reference `[1.481490, 1.672045, 1.524825] ms`、candidate `[0.478665, 0.485885, 0.484020] ms`；正式中位数为 `1.524825 → 0.484020 ms`，即 `3.1503450270657824x`、wall 改善 `68.25734026854723%`。device time 从 `481.5474` 降至 `320.7216 us/call`，下降 `33.3967%`；candidate 侧 eager component kernel 数为 0，且没有 candidate-added synchronization。wall 的 `68.26%` 改善大于 device-time 的 `33.40%` 改善，说明 launch / runtime 与 global intermediate traffic 的消除都是收益来源，而不仅是数学运算变快。

accepted record 还覆盖五次 benchmark accuracy、输出 shape/dtype/layout/device 契约、输入不变性、重复执行稳定性与独立 allocation、per-matrix independence、输出 non-alias、固定参数、精确 Sinkhorn 操作顺序和 current-stream 行为。所有 promotion 前置条件通过，结果超过 5% 采纳门槛，verdict 为 `terminal_result=accepted`、high confidence。candidate wall 与 AI Core summed duration 的差值 `163.2984 us`（约 candidate wall 的 `33.74%`）只是一项诊断，不应被直接标记为可完全消除的 host overhead 或承诺为后续收益。

**结论**：该收益来自完整迭代链的 launch 与中间 tensor 消除，而不是单个数学 primitive 的替换。原生实现必须同时闭合 Python loader、CMake target、生成库名、launcher ABI、输入约束和 stream ownership；缺少其中任一项都不能称为可交付实现。

## 跨算子经验

1. **先证明完整算子语义，再看性能。** load、export、局部 probe、all-invalid 和 nominal correctness 是不同层次的证据；不能用低层成功替代完整语义。
2. **ABI 是实现的一部分。** 对原生候选，Python ctypes 签名、CMake target、C++ entry、共享库名、launcher symbol 和 caller current stream 必须相互对应；源码存在但调用边界不闭合不能交付。
3. **以端到端成本而不是“是否原生”判断 ROI。** `index_topk` 的 native candidate 正确却为 `0.608x`，说明 full-size materialization、外部 reduction、cast、转移和同步可超过局部 kernel 收益。
4. **固定状态应在初始化时处理。** 当 mask、有效长度或其他约束只依赖静态配置时，预计算可以减少 forward 热路径开销，同时避免不必要的 dtype 转换。
5. **融合适用于被 launch 与临时张量主导的完整链。** `sinkhorn_normalize` 的 59→1 融合同时降低 wall 和 device time；只有掌握完整迭代/归约边界时，融合才是可验证的优化方向。
6. **负结果也应以证据保留。** `sparse_attn` Round 006 和 `index_topk` native path 分别说明“未通过 nominal correctness”与“正确但端到端退化”是两种不同的停止原因，均不应被包装为成功实现。
7. **canonical 与实验候选严格分离。** 当前仓库只保留可引用源码；被拒绝候选的机制、测量和 verdict 保留在证据快照中，供审阅而不造成错误复用。
8. **Profiler 与 wall-time 需要分开解释。** wall、summed device task duration 与 runtime / launch overhead 是不同指标；device duration 的差值既不能自动解释为可消除 host overhead，也不能直接外推 wall 改善。
9. **环境结论不得外推。** 当前原生记录依赖目标 Ascend runtime、CANN、编译器和输入约束。本文不把单一 shape 的结果推广为其他设备、动态 shape 或其他算子的 capability claim。

## 运行环境与适用边界

归档的 runtime fingerprint 为 `npu:0 / Ascend910B4`、target `ascend910b`、architecture `ascend-910b4`、Python `/usr/local/python3.11.15/bin/python3`、PyTorch `2.7.1+cpu`、`torch_npu 2.7.1.post4` 与 CANN `9.0.0`。Sinkhorn 的原生路径还依赖 CMake、Ascend C CMake integration、`kernel_operator.h` 与匹配的 Ascend910B4 toolchain。

harness-facing `get_inputs` 中的 `device="cuda"` 是由 harness 搬运 / 重写到 NPU 的占位约定，不代表这些实现是 CUDA kernel。所有性能、build 与 capability 事实只归属于上述 runtime fingerprint；不得从某个算子的 exact probe 推断其他算子的完整 native capability。

## 证据溯源

以下文件逐字节复制自闭合归档 `kernelswift-track2-ascendc-round006-closed.tar.gz`，其整体 SHA-256 为 `eaf64ea5f5c926dcb5e5d1c4d0cc974f486a87b939c72a715ee05d1081d35fea`。归档 member 名保留了原有 `rounds/` 层级；这是 provenance 文本，不代表紧凑树恢复了该目录。

| 当前快照 | 归档 member | 用途 | SHA-256 |
|---|---|---|---|
| [`sparse_attn/evidence/baseline_report.md`](sparse_attn/evidence/baseline_report.md) | `kernelswift-track2-ascendc-round006/./kernels/track2-clike/sparse_attn/ascendc/rounds/report_000.md` | baseline qualification 与测量 | `fc9fb954e28fb8bb7a248431350257d5f6968ed313c43b54242c831f407f058c` |
| [`sparse_attn/evidence/candidate_failure.md`](sparse_attn/evidence/candidate_failure.md) | `kernelswift-track2-ascendc-round006/./kernels/track2-clike/sparse_attn/ascendc/rounds/report_006.md` | Round 006 nominal failure 边界 | `432fb867d96b8783ecaa26f4b6751ada01979eb43e35f424a57625844b647e29` |
| [`sparse_attn/evidence/candidate_failure_verdict.json`](sparse_attn/evidence/candidate_failure_verdict.json) | `kernelswift-track2-ascendc-round006/./kernels/track2-clike/sparse_attn/ascendc/rounds/verdict_006.json` | machine-readable failure verdict | `b3f2fadf8fc75dd382ee3006167123170dc2475d43ec32c18dace5e90e99dee1` |
| [`index_topk/evidence.md`](index_topk/evidence.md) | `kernelswift-track2-ascendc-round006/./kernels/track2-clike/index_topk/README.md` | canonical / rejected path 的详细测量与机制 | `9c42092b65d632452a3897a0283713196c747a3cbf2e0598c8f0af111268dcdc` |
| [`sinkhorn_normalize/evidence/accepted_report.md`](sinkhorn_normalize/evidence/accepted_report.md) | `kernelswift-track2-ascendc-round006/./kernels/track2-clike/sinkhorn_normalize/ascendc/rounds/report_001.md` | accepted native result | `670ff0d069a73b5c670a26fc2512c0597c081319168fee86e27e809fd612d3aa` |
| [`sinkhorn_normalize/evidence/accepted_verdict.json`](sinkhorn_normalize/evidence/accepted_verdict.json) | `kernelswift-track2-ascendc-round006/./kernels/track2-clike/sinkhorn_normalize/ascendc/rounds/verdict_001.json` | machine-readable accepted verdict | `9a754e32a56c73b7fc1bfb24d66c8890ee21e84fb88ff1c2278b2acbc4549af0` |

## 未声明的结论

- 本次整理不声明新的 Ascend 硬件测量、编译结果或 correctness 结果。
- 不声明 `sparse_attn` Round 006 有任何可引用性能值；它在 timing 前已因 nominal correctness 失败终止。
- 不声明 `index_topk` 的 eager canonical 是原生 Ascend C，也不将其 `1.059x` 外推为 native capability。
- 不声明单一输入 shape、toolchain 或设备上的经验对其他设备、shape、runtime 版本或算子同样成立。

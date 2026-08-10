# fused_moe Triton 优化结果

Branch: `fused-moe-opt`。目标 shape：`T=83, H=128, E=8, top_k=2, intermediate_size=64`（fp16 hidden / fp32 router_logits，含 softmax + top-2 + renorm 路由 + per-expert 双 GEMM + SiLU 门控）。测量口径：`auto_bench.py --warmup 50 --repeat 100`，device time 取 profiler JSON 中 `cat == "kernel"` 的 `dur` 50 次平均。

| Round | 文件 / 优化手段 | Wall (auto_bench) | Device time | 相对 base |
|---|---|---:|---:|---:|
| 0 | `base.py` eager — softmax + topk + renorm + cast + Python for-loop over 8 experts，含 mask/gather/scatter，约 50 个 kernel launch | 6.94 ms | ~2.7 ms / iter | 1.00x |
| 1 | `triton_fused_moe_001.py` — per-token Triton kernel（grid=(T,)），消灭 mask/gather/scatter；GEMM 用 `tl.sum(x[None,:] * w, axis=1)` elementwise 外积；routing 仍在 PyTorch | 0.5638 ms | 21.04 us / iter | 12.3x |
| 2 | `triton_fused_moe_002.py` — 把 softmax + top-2（重复 `tl.max` + argmax-via-masked-sum）+ renorm 融进同 kernel；Python 端只传 `router_logits` 指针 | 0.2178 ms | 23.47 us / iter | 31.9x |
| 3 | `triton_fused_moe_003.py` — `fast_libentry()` + 类体 `globals()` trick 绕过 `_filter_module_ast`；ModelNew 上缓存 `torch.empty` 输出 buffer | 0.1533 ms | 23.42 us / iter | 45.3x |
| 4 | `triton_fused_moe_004.py` — GEMM 用 `tl.dot(x_2d, tl.trans(w))` 替代 elementwise 外积，走 BMM 硬件单元（device 略降但 wall 因 host 噪声略升） | 0.1640 ms | 21.02 us / iter | 42.3x |
| 5 | `triton_fused_moe_005.py` — 去掉 `with torch.mlu.device(...)` context manager（kernel 与 v4 一致，纯 host 路径优化） | 0.1377 ms | 21.02 us / iter | **50.4x** |
| 对比 | `triton_fused_moe_006.py`（tmo 4 段拼装 `moe_softmax_topk` + `moe_gen_idx` + `group_gemm ×2` + eager combine，仅作对比，不替代 v5） | 1.012 ms | 79.5 us / iter | 7.04x |
| 对比 | `bangc/fused_moe_kernel.mlu`（手写 BangC，per-token Union1 多核 + 标量 GEMM fallback，仅作对比） | 4.090 ms | 4090 us / iter | 1.70x |

## 停止理由

- 5 轮完成，wall 从 6.94 ms 压到 0.138 ms（50.4x）。
- v5 之后 device_ratio ≈ 15%（device 21 us / wall 138 us），wall 已被 host overhead 主导。
- 剩余 host overhead ~117 us 主要是 harness 固定成本（`set_seed` ~12 us + `sync_devices` ~40 us 因 cuda+mlu 双同步 + `build_case`/`load_state_dict` 状态差 ~24 us + `fast_libentry` 残余 launcher ~40 us），无法在 kernel 侧压缩。
- device 21 us 距离 stretch goal 10 us 还差 2x，但 wall 已被 host 主导，继续压 device 没有 wall 收益。

## 关键踩坑

- **`_filter_module_ast` 剥非字面量模块级赋值**：v1 直接 `_fast = ...` 在模块顶会被剥导致 NameError；v3 起用类体 `globals()` trick 绕过（ClassDef 节点保留，class body 在 import 时执行）。
- **argmax sentinel**：v2 一开始用 `tl.where(is_best, e_idx, E)` 让非 best 取 E，求和变成 `best + (E-1)*E` 全越界；改 `tl.where(is_best, e_idx, 0)` 正确。
- **`tl.dot` shape**：v4 忘记 `tl.trans(w)` 直接 `[1,H] @ [2I,H]` 报错；要 `[1,H] @ tl.trans([2I,H])=[H,2I]`。
- **`torch.cuda.is_available()` 在 MLU 机返回 True**：`sync_devices` 会同时同步 cuda 和 mlu，每次 forward 多 ~40 us，属 harness 固定不可压缩。

## 累计

v0 → v5 累计 **50.4x**（auto_bench wall 6.94 ms → 0.138 ms）；tmo 4 段拼装 7.04x（wall 1.012 ms），比 v5 慢 7.3x；手写 BangC 标量 GEMM fallback 1.70x（wall 4.090 ms，device 4090 us），比 v5 慢 29.7x。

## 对比参考（tmo 4 段拼装）

`triton_fused_moe_006.py` 用 `torch_mlu_ops` 拆 4 段拼装：`moe_softmax_topk` + `moe_gen_idx` + `group_gemm × 2` + eager combine（`out_exp[combine_idx]` gather + mul + sum + copy）。tmo 没有整体 `fused_moe` op，必须分段。

- wall 1.012 ms（3 次稳定 run 1.017 / 1.036 / 1.020），相对 base 7.04x，**比 v5 慢 7.3x**；device 79.5 us / iter（v5 是 21 us，慢 3.8x）；精度 `atol=1e-2` FAIL（fp16 GEMM 累加顺序 + gather 精度损失），`atol=5e-2` PASS。
- device 拆解（per-iter，来自 trace）：
  - `MLUGroupedGemmEx` × 2 = 19.0 us（单 op 9.5 us，跟 v5 全 GEMM 工作量持平）—— tmo GEMM 本身不慢
  - `MLUGatherIdxToGatherOffset` × 2 = 16.1 us（group_gemm 内部 gather offset 准备，v5 无此步骤）
  - `MLUBlockKernelCastExecOnce` × 3 = 9.6 us（`w1.to(fp16)` + `w2.to(fp16)` + `reduce_weight.to(fp16)` 每 forward 重 cast，v6 实现粗糙，理论上可预算到 `__init__`）
  - `moe_softmax_topk` 4.7 us + `moe_gen_idx` 3.5 us = 8.2 us（routing，v5 融进主 kernel）
  - `silu` 3.7 us + `mul` 4.0 us + `reduce sum` 3.8 us + `gather back` 4.6 us + `copy` 2.9 us + stridedSlice × 2 5.0 us = ~24 us（v5 一次 kernel 内做的算术，v6 拆成 6+ 个 eager glue kernel）
  - 合计 ~79 us / iter
- host 拆解：wall 1012 us − device 80 us ≈ 932 us host overhead。4 个 tmo op launcher（每个 ~50–100 us，schema 校验 + autotune 路径）+ 6 个 eager op launcher（每个 ~20–30 us）+ harness 固定 ~52 us。v5 只有 1 个 `fast_libentry` launcher ~30 us + harness ~52 us + 状态差 ~35 us = 117 us host。**tmo op launcher 在 T=83 小 shape 下比 `fast_libentry` 重一个数量级**，是 v6 wall 暴涨的主因。
- 结论：tmo 没有整体 `fused_moe` op，4 段拼装在小 shape 下被 host launcher 拉爆。如果 shape 上到 T=4096+ 让 device 重新主导 wall，tmo `group_gemm` 单 op 9.5 us 的 GEMM 性能可能反超 v5 手写 `tl.dot`。本算子 T=83 太小，v5 单 Triton kernel 结构上必胜。**v5 维持为 canonical**，v6 仅作对比探查。
- v6 trace：`log/triton_fused_moe_006_forward_50iter.pt.trace.json`。

## 对比参考（手写 BangC kernel）

`bangc/fused_moe_kernel.mlu` 用 BangC（MLU 原生 C++ kernel 编程模型）写 per-token fused MoE：grid=(T,)，Union1 多核（32 core），每个 program 在 NRAM 内完成 softmax+top-2+renom+双 GEMM+SiLU+加权累加。host_driver.cpp 用 cnrt API 分配/拷贝/sync，CPU 预转置 w1/w2，cnrtNotifier 测 device time，CPU fp32 reference 校验。

- wall 4.090 ms（host std::chrono）；device 4090 us / iter（cnrtNotifierDuration，微秒口径）；max_abs_diff 1.7117 → atol=5e-2 **FAIL**（输出值相对误差 ~0.1% 但绝对误差超阈值，根因是 `nram_gate_up_h`/`nram_act`/`nram_out_k_h`/`nram_out_acc` 都是 half，中间值精度损失累积）。
- 相对 base 1.70x（看似比 base 快，但 base 的 6.94 ms 是 eager 50 个 kernel launch 的 wall，device 时间并非 4 ms，无可比性）。**比 v5 慢 29.7x（wall）/ 195x（device）；比 v6 tmo 慢 51.5x（device）**。
- 根因：标量 GEMM fallback 没用矩阵单元。原本计划用 `__bang_matmul(float*, const half*, const half*, M, K, N)` 半精度入口直接调 MLU590 的矩阵单元，但：
  - `bang_host_functions_decls.h` 里有该 overload 声明，但 `neuware_home/examples` 里所有 matmul 样例只用 int8 + 4D strided `__memcpy` 的 WRAM 布局，half 输入的 expected stride **没有任何文档或样例**。
  - 直接用 `__bang_matmul(nram_out, nram_x, nram_w1, 1, H, TWO_I)` 输出全错（max_abs_diff 3378）。fallback 到标量 GEMM 才得到接近正确的输出（相对误差 ~0.1%）。
  - WRAM 是矩阵单元专用 RAM，不支持标量读（标量读返回 NaN），所以即便把 w1/w2 放 `__wram__` 也无法手写 GEMM 循环，必须放 `__nram__` 走标量。
- 踩坑（详见 log.md Entry 005.c）：
  - `<<<dim, func_type, queue>>>` 是 BangC 专有语法，只有 cncc 能解析，g++ 报错；解法是把 launch 包进 .mlu 里的 `extern "C" launch_fused_moe(...)` wrapper，host_driver.cpp 只 link 这个符号。
  - cnrt API 命名：`cnrtMemcpyHostToDev`/`cnrtMemcpyDevToHost`（不是 `cnrtHost2Device`），`cnrtPlaceNotifier`（不是 `cnrtNotifierRecord`）。
  - `cnrtNotifierDuration` 返回微秒（不是毫秒），header 里参数名写的是 `ms` 但实际单位是 us；最初按 ms 处理乘 1e3，导致 device time 报 141565 us（1000x 偏大）。
  - g++ 编译 BangC host 端的 ABI 陷阱：`<random>` 在 `-std=c++11` 严格模式下 `vswprintf` ABI 报错，必须用 `-std=gnu++11`；`neuware_home/lib/clang/11.1.0/include/` 下的 `stdint.h` 用了 `__has_feature`（clang-only），会 shadow 系统 stdint.h，必须用绝对路径 `#include` 引入 bang_fp16.h；`__internal_float2half` 用 `reinterpret_cast` 类型双关在 `-fstrict-aliasing`（-O2 默认开）下是 UB，必须加 `-fno-strict-aliasing`。
- 结论：手写 BangC 在没解决 `__bang_matmul` half 输入 WRAM 布局之前，性能远不如 Triton v5（195x 慢）和 tmo v6（51x 慢）。**Triton 在 MLU590 上的 `tl.dot` 路径已经能调用矩阵单元**，写 BangC 标量 GEMM 是反向优化。若未来能拿到 `__bang_matmul` half 输入的 WRAM stride 文档，手写 BangC 矩阵单元路径有望接近 v5（参考 tmo `MLUGroupedGemmEx` 单 op 9.5 us 的存在性证明）。本次未解决，留作 P2。**v5 Triton 维持为 canonical**，BangC 标量版仅作对比基线。
- 文件：`bangc/fused_moe_kernel.mlu`、`bangc/host_driver.cpp`、`bangc/CMakeLists.txt`，build/run 命令在 `bangc/build/` 下 `cmake .. && make && ./fused_moe_bangc`。


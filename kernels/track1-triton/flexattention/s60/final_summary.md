# FlexAttention S60 (GCU) 优化结果

Branch: `kernel-opt/flexattention-s60`。目标 shape：`T=83, H=8, D=64, Kv=8`（GQA ratio 1, fp16, causal SDPA）。测量口径：`auto_bench.py --warmup 50 --repeat 100`，wall time 取 median；GCU profiler 不提供 device kernel duration（`device_time_available=false`），仅保留 `runtime_launch_*` 诊断证据。

## 结论：measurement-bound，无优化空间

| Round | 文件 / 手段 | Wall (auto_bench) | Runtime launches/call | 结论 |
|---|---|---:|---:|---|
| 0 | `base.py` eager — `F.scaled_dot_product_attention` | 0.269216 ms | 1.0 | 1.00x baseline |
| 1 | `decision_001.md` — abort | - | - | 无 >=5% 可证伪干预 |

## 根因分析

flexattention 在 s60 上**没有优化空间**,与 MLU 的 7.08x 结论相反:

1. **eager SDPA 已是单 kernel**：GCU 的 `F.scaled_dot_product_attention` 被 CNNL
   融合成单个 flash-attention 库 kernel（1 个 `topsLaunchKernel`/call）。MLU 靠
   "22 个 eager kernel → 1 个 Triton kernel" 拿 7.08x，但 GCU 上这一步已经被库
   完成，没有 kernel 数可减。
2. **手写 Triton 打不过库实现**：本地同 regime 探针实测，手写 causal-SDPA
   kernel（每 `(token, head)` 一个 program，`tl.dot` 做 QK^T/AV）正确
   （max_abs_diff 1.95e-3），但 device 执行 ~100x 慢于 CNNL flash attention
   （`forward+sync` ~22 ms vs ~0.15 ms）——664 个 tiny program 的 `[1,64]x[64,128]`
   dot 无法匹配库的 fused 实现；要达到库级别需 online-softmax 分块，而 GCU 上
   fp16 `tl.dot` 分块性能未探明，且 launch 数仍 1→1。
3. **host 成本 harness 固定**：wall 0.269 ms 中，runtime launch 仅 ~10.5 us/call，
   其余是 harness 固定成本（`set_seed` + `torch.gcu.synchronize()` ~85 us + 其它）。
   base.py 与 harness 不可变，host 侧无合法优化空间；groupedtopk s60 已实测同类
   host 优化（output cache / device-context）仅 +2.06%，低于 5% 阈值。

## 与 groupedtopk s60 的共性

两个算子在 s60 上得出同一结论：GCU runtime 上 eager 路径已被 CNNL 库优化，
device 侧无手写 Triton 的收益空间，wall 被 harness 固定 host 成本主导。这属于
**测量边界（measurement-bound）**，不是 GCU 硬件或 Triton-GCU 的缺陷，而是当前
runtime/profile 无法为手写 Triton 提供超越库实现的证据。

## 停止理由

- 用户显式停止（`user-intervention`）。
- Round 1 abort 判定无 >=5% 可证伪干预；三个方向（kernel 融合 / 手写 Triton /
  host 优化）均被 Phase 0 trace + 本地探针证据否定。

## 累计

baseline 保持 `baseline_adapter.py`（0.269216 ms），无加速比。flexattention 在
s60 的 canonical 结论为 **measurement-bound，无可优化空间**。

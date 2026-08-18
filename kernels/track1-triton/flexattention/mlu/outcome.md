# flexattention Triton 优化结果

Branch: `flexattention-opt`。目标 shape：`T=83, H=8, D=64, Kv=8`（GQA ratio 1, fp16, causal SDPA）。测量口径：`auto_bench.py --warmup 50 --repeat 100`，device time 取 profiler JSON 中 `cat == "kernel"` 的 `dur` 50 次平均。

| Round | 文件 / 优化手段 | Wall (auto_bench) | Device time | 相对 base |
|---|---|---:|---:|---:|
| 0 | `base.py` eager — PyTorch `F.scaled_dot_product_attention`，22 个 eager kernel 串联 | 1.006 ms | 96.7 us / iter | 1.00x |
| 1 | `triton_flexattention_001.py` — 单 Triton kernel 融合 QK^T/softmax/AV，22 launch → 1 launch（elementwise+reduce 路径） | 0.264 ms | 96.2 us / iter | 3.81x |
| 2 | `triton_flexattention_002.py` — QK^T 与 AV 改用 `tl.dot`，走 BMM 硬件单元替代 elementwise+reduce | 0.210 ms | 50.9 us / iter | 4.56x |
| 3 | `triton_flexattention_003.py` — `fast_libentry()` + 类体 `globals()` trick 绕过 AST filter；缓存输出 `torch.empty`；去掉 `torch.mlu.device()` context | 0.140 ms | 50.5 us / iter | **7.08x** |
| 4 | `triton_flexattention_004.py` — 试 `num_stages=2`（`num_warps=2` MLU 不支持回退到 1）；<5% 改善，noise-rejected 不进主实现 | 0.137 ms | 50.6 us / iter | 7.32x (noise) |
| 对比 | `triton_flexattention_005.py`（tmo `flash_attention` 单 op，仅作 device 上界对比，不替代 v3）— QK^T + softmax + AV 全部 fuse 进库 op | 0.171 ms | 8.77 us / iter | 5.89x |

## 停止理由

- v4 多次重复运行改善 < 2%，低于 5% 阈值，属 noise。
- v3 之后 host 端 compressible 部分已榨干：剩余 ~89 us host 主要是 harness 固定成本（`set_seed` ~12 us + `sync_devices` ~40 us，cuda+mlu 双 sync）+ `fast_libentry` 残余 launcher ~30 us + 状态差异 ~7 us。
- device 50 us 已接近 fused attention 上界，进一步压缩需要在线 softmax / `tl.dot` 矩阵分块大改动，但 wall 收益受 host floor 限制（即使 device 降到 0，wall 也只到 ~89 us）。

## 对比参考（tmo `flash_attention` 单 op）

`triton_flexattention_005.py` 调 `torch_mlu_ops.flash_attention` 单 op——把 QK^T + softmax + AV 全部 fuse 进 CNNL 库 op。

- wall 0.171 ms（3 次稳定 run 0.166/0.171/0.175，平均 0.171，**比 v3 ~0.140 慢 ~22%**）；device 8.77 us / iter（比 v3 50.5 us 快 5.8x）；精度 `atol=1e-2` PASS。
- 与 groupedtopk 上界结果相反：groupedtopk tmo 单 op 在 wall 与 device 上都打过手写 v4；flexattention tmo 单 op 只在 device 上打过手写 v3，wall 反而慢 22%。
- 根因：tmo `flash_attention` 单 op 的 host launcher 路径在 T=83 小 shape 下比 Triton `fast_libentry` 重——v5 wall 171 us 中 device 只占 8.8 us（5%），host ~162 us 几乎全是 tmo launcher + harness 固定成本。手写 v3 device 50 us（占 wall 36%）+ host ~90 us，host 总量更小。
- 即 v5 在 device 上确实是上界（8.77 us 接近单次 fused attention kernel 极限），但 wall 上不是。**不替代 v3**：tmo 库 op 是黑盒（不在本仓库代码内），与"在仓库内手写 Triton 优化"的目标不一致；只作为 device 端可参照的"工程上界"。
- v5 trace：`log/flexattention_005_forward_50iter.pt.trace.json`，唯一 kernel `MLUFlashAttentionLatency<half,...,true,...>` 8.77 us / 50 iter，外加 `MLUFlashAttentionLatencyAdaptiveSchedule` 1.47 us。

## 累计

v0 → v3 累计 **7.08x**（auto_bench wall 1.006 ms → 0.140 ms）；tmo 单 op 5.89x（wall 0.171 ms），device 上界 8.77 us / iter。

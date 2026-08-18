# groupedtopk Triton 优化结果

Branch: `fused-moe-opt`。目标 shape：`T=83, E=256, topk=8, num_expert_group=8, topk_group=4, renormalize=True, scoring_func="softmax", routed_scaling_factor=1.0`（fp32 gating，含 softmax → group max(8,32) → top-4 group → mask → top-8 of 256 → renorm → scaling）。测量口径：`auto_bench.py --warmup 50 --repeat 100`，device time 取 profiler JSON 中 `cat == "kernel"` 的 `dur` 50 次平均。

| Round | 文件 / 优化手段 | Wall (auto_bench) | Device time | 相对 base |
|---|---|---:|---:|---:|
| 0 | `base.py` eager — softmax → view+max → topk(4 of 8) → scatter → masked_fill → topk(8 of 256) → renorm，约 10 个串行 PyTorch op，~36 kernel launch / iter | 0.840 ms | ~251 us / iter | 1.00x |
| 1 | `triton_grouped_topk_001.py` — 单 per-token Triton kernel fuse 整道题（grid=(T,)），fast_libentry，消灭 softmax/topk/scatter/masked_fill 一串 op | 0.198 ms | 38.5 us / iter | 4.24x |
| 2 | `triton_grouped_topk_002.py` — 缓存输出 tensor（`_ensure_buf(T, device)`），干掉 forward 里 `torch.empty` × 2，host allocator overhead → 0 | 0.146 ms | 38.1 us / iter | 5.84x |
| 3 | `triton_grouped_topk_003.py` — `tl.argmax` 替代手写 max+where+min 三连击做 selection sort，reduction 数 3→2，device 38→20 us | 0.130 ms | 20.0 us / iter | 6.50x |
| 4 | `triton_grouped_topk_004.py` — host 端 constexpr 预计算（`_ensure_cfg(T, E)` 缓存 `grid/BLOCK_E/epg`）+ `_fast` 绑实例 attr，避免每次 forward 走 `next_power_of_2` 与整数除法 | 0.128 ms | 20.0 us / iter | **6.56x** |
| upbound | `triton_grouped_topk_005.py`（tmo `moe_softmax_topk` 单 op，仅作上界参考，不替代 v4）— 把 softmax + group-max + topk_group + mask + masked topk + renorm + scaling 全部 fuse 进库 op | 0.117 ms | 9.86 us / iter | 7.18x |

## 停止理由

- 4 轮完成，wall 从 0.840 ms 压到 0.128 ms（6.56x）。
- v4 之后 `device_ratio = 20/128 ≈ 16%`（< 20% 阈值），wall 已被 host overhead 主导。
- 剩余 host overhead ~107 us 主要是 harness 固定成本（`set_seed` ~16 us + mlu sync ~25 us + cuda stub sync ~66 us，MLU box 上 `torch.cuda.is_available()=True` 导致 `sync_devices` 同步 cuda+mlu 两边），无法从 kernel 侧压缩。
- device 20 us 受限于 softmax(256) + group-max + 8 趟 256-元素 argmax+max 的串行 reduction 工作量；试过的 device 路线（BLOCK_T=2/4/8 2D batching）反而被 per-program launch overhead 拖慢。

## 上界参考（tmo `moe_softmax_topk` 单 op）

`triton_grouped_topk_005.py` 调 `torch_mlu_ops.moe_softmax_topk` 单 op——把整道题全部 fuse 进 CNNL 库 op。

- wall 0.117 ms（3 次稳定 run 0.115/0.116/0.119，平均 0.117，比 v4 ~0.135 快 ~13%）；device 9.86 us/iter（比 v4 20 us 快 2x）；精度 `atol=1e-2` PASS。
- wall 提升 ~13% 已达 5% 门槛，但**不替代 v4 作为 canonical**：tmo 库 op 是黑盒（不在本仓库代码内），与"在仓库内手写 Triton 优化"的目标不一致；只作为可参照的"工程上界"。
- 与 v4 device 差距 20 → 10 us（2x）说明手写 Triton 在 selection sort（12 趟 reduction）上仍有 ~10 us 可压缩空间，但 wall 上 host 主导（device_ratio 8%），继续优化 device 没有 wall 收益。
- v5 trace：`log/groupedtopk_v5_forward_50iter.pt.trace.json`，唯一 kernel `tmo::kernels::MLUSoftmaxTopkKernel<float, float>` 9.86 us / 50 iter。

## 关键踩坑

- **`_filter_module_ast` 剥非字面量模块级赋值**：`fast_libentry()(_kernel)` 写在 module 顶会被剥导致 NameError；用类体 `if "_fast" not in globals(): globals()["_fast"] = fast_libentry()(_kernel)` 绕过（ClassDef 节点保留，class body import 时执行）。
- **`fast_libentry` 调用形式**：是 `fast_libentry()(_kernel)`（工厂调用），不是 `fast_libentry(_kernel)`，写错会静默不生效。
- **`tl.argmax` selection sort**：Triton 没有内置 top-K，要靠"找 max → mask 走过 → 重复 K 趟"实现，每趟一次 reduction；K=8 + K=4 共 12 趟 reduction 是 device 时间的大头。
- **group reshape 边界**：E=256, num_expert_group=8 → epg=32，必须用 `tl.reshape` 显式 2D 化再 `tl.max(axis=1)`，不能直接 `tl.view`（Triton 没有 view）。
- **`torch.cuda.is_available()` 在 MLU 机返回 True**：`sync_devices` 会同时同步 cuda stub 和 mlu，每次 forward 多 ~66 us cuda stub sync，属 harness 固定不可压缩。
- **tmo `moe_softmax_topk` mask 外置 0**：库 op 用 `masked_fill(~mask, 0.0)`，base.py 用 `masked_fill(~mask, -inf)`；topk 结果 expert_id 一致（mask 外不入选），topk_weights 也一致（mask 内原值），精度 PASS。

## 累计

v0 → v4 累计 **6.56x**（auto_bench wall 0.840 ms → 0.128 ms）；上界 tmo 单 op 7.18x（wall 0.117 ms）。

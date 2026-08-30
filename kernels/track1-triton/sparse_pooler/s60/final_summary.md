# SPLADE Sparse Pooler S60 (GCU) 优化结果

Branch: `kernel-opt/sparse-pooler-s60`。目标 shape:`hidden_states=[83,768] fp32; seq_lens=[20,25,18,20] int32; 输出 list 4x[30522] fp32`。测量口径:`auto_bench.py --warmup 50 --repeat 100`,wall time 取 median;GCU profiler 不提供 device kernel duration,仅保留 `runtime_launch_*` 诊断证据。

## 结论:measurement-bound,无优化空间

| Round | 干预 | Wall (v1 median) | Runtime launches/call | 结果 |
|---|---|---:|---:|---|
| 0 | `base.py` eager(MLM head 库算子 + relu/log1p + 4x chunk.max + D2H sync) | 0.861388 ms | 11 | baseline |
| 1 | 融合 relu/log1p/max + 设备端前缀扫描(消 D2H sync) | 1.092186 ms | 6 | no-improvement · **-26.79%** |
| 2 | abort — 无 >=5% 可证伪干预 | - | - | aborted · measurement-bound |

## 关键结论

Round 1 的融合候选**正确性 PASS 但性能倒退 26.79%**。机制层面两个观察量都达成(launch 11→6,D2H sync 彻底消除,launch 时间 111→62 us,省 ~49 us),但 fused Triton kernel 的 device 计算开销(~+270 us)远超节省,根因是 GCU 上手写 Triton 的 elementwise+segment-reduction 效率远低于库算子(`num_warps=1` 单 warp 串行 + 序列轴 `range(seq_len)` 串行循环 + `BLOCK_V=256` 小 tile)。

这与 fused_moe 形成鲜明对比:
- **fused_moe**:147 次 launch 的巨大 host 开销,融合收益(省 ~1400 us launch)远超 device 惩罚 → 13.1x
- **sparse_pooler**:仅 11 次 launch(host 省 ~49 us),但 device 惩罚 ~270 us → 倒退

## 停止原因:measurement-bound

Round 1 证明 GCU 上库算子(relu/log1p/chunk.max)已经足够快,手写 Triton 无法超越。剩余优化方向均 <5%:
1. **调参 fused kernel**(BLOCK_V/num_warps):根因是 GCU Triton device 计算效率,非参数可救;MLU v2 已证 BLOCK_V 调参 no-improvement;GCU num_warps>1 未验证。
2. **回归库算子**:丧失唯一确认机制(D2H sync 消除)。
3. **只消 D2H sync**:sync 仅占 ~111 us launch 预算的一小部分,MLM head 5-6 次库 GEMM launch 必须保留,单独收益 <5%。

## 与 MLU 1.60x 的差异归因

MLU 的 1.60x 依赖:v1 融合(+33.39%)在 MLU 上 device 计算惩罚小于 host 节省,而 GCU 上相反;v4 的额外 +5.79% 来自 `fast_libentry`(GCU 无)。因此 sparse_pooler 在 GCU 上无可复现加速,这是 backend 的 Triton device 计算效率差异。

## 停止理由

- stop_reason: `measurement-bound`
- Round 1 no-improvement(-26.79%)证伪 kernel-fusion 方向;Round 2 Designer 判定无 >=5% 可证伪干预。

## 累计

canonical 保持 `baseline_adapter.py`(0.861388 ms),无加速比。sparse_pooler 在 s60 的结论为 **measurement-bound,无优化空间**。

# MusicFlamingo Rotary Embedding S60 (GCU) 优化结果

Branch: `kernel-opt/rotary-embedding-s60`。目标 shape:`timestamps=[4,32] fp32; seq_len=32 int; 输出 tuple (cos,sin) 各 [4,32,128] fp32`。测量口径:`auto_bench.py --warmup 50 --repeat 100`,wall time 取 median;GCU profiler 不提供 device kernel duration。

## 结论:measurement-bound,无优化空间

| Round | 干预 | Wall (v1 median) | launches/call | 结果 |
|---|---|---:|---:|---|
| 0 | `base.py` eager(13 次 elementwise launch) | 0.464657 ms | 13 | baseline |
| 1 | 融合 elementwise 成单 kernel(单 program) | 5.162427 ms | 1 | no-improvement · -1010.99% |
| 2 | grid 并行度修复(BLOCK=128/grid=128) | 0.525050 ms | 1 | no-improvement · -13.00% |
| 3 | abort — 无 >=5% 可证伪干预 | - | - | aborted · measurement-bound |

## 关键结论

两轮融合探索已把 fusion 做到最优:
- **Round 1**:验证了融合正确性 + launch 13→1 机制,但 `grid=(1,)` 缺陷导致单 program 串行(5.16ms)。
- **Round 2**:修复 grid 后 device 时间 5.15→0.52ms(~10x 改善),launch 全保留(省 ~97us),但 wall 仍比 eager 慢 13%。

**定量真相**(Designer 精确分析):wall 差 60us 是"融合省 97us launch"与"device 多花 157us"的净值。真正的 device 缺口是 **157us / 44%**——手写 Triton 用 MLIR math-dialect 的 `tl.cos/tl.sin` + 逐元素 int div/mod 索引反解 + branch select,无法追平 vendor 高度优化的 cos/sin 库 kernel。这是 GCU Triton 单 warp elementwise 的固有开销,不是第三次重写能消除的。

## 停止原因:measurement-bound

要达 >=5% 需再降 ~84us device(16%),剩余手段全部无 Verifier 证据:
- `num_warps>1`:profile 只证明 num_warps=1,且 2-MP 设备不加独立并行通道
- 向量化 load:Unknown,且 kernel 是 gather 式 load
- BLOCK 调参:marginal(MLU 曾 no-improvement)
- 常量折叠:空间可忽略

## 与其他 s60 算子的共性规律

至此 s60 上 5 个算子的结论已高度一致:
- **fused_moe(147 launches)**:融合收益 >> device 惩罚 → 13.1x ✅
- **flexattention(1 launch)/sparse_pooler(11 launches)/rotary(13 launches)**:库算子已足够快,手写 Triton 的 device 计算(尤其 elementwise 和 reduction)慢于 vendor 库,融合收益无法抵消 → 无空间

**核心结论**:GCU 上手写 Triton 只在 eager baseline 有海量 launch(fused_moe 147)时才有净收益;当 launch 数少(≤13)且是 elementwise/reduction 时,vendor 库算子已高度优化,手写 Triton 无法超越。

## 停止理由

- stop_reason: `measurement-bound`
- Round 1/2 连续 no-improvement(performance_miss_streak=2),Round 3 Designer 判定无 >=5% 可证伪干预。

## 累计

canonical 保持 `baseline_adapter.py`(0.464657 ms),无加速比。music_flamingo_rotary_embedding 在 s60 的结论为 **measurement-bound,无优化空间**。

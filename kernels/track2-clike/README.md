# 赛道二：C-like 算子优化

本目录保留赛道二三个算子的参考实现、当前 canonical 源码、完整结果总结与最小原始证据包。性能数据均来自已验证归档中的 `auto_bench.py` 完整算子 wall-time 记录；只有同时满足正确性、语义 guardrail 与性能采纳条件的实现才会成为 canonical。

## 结果总览

| task | 算子 | 当前 canonical | 记录结果 | 关键证据 |
|---|---|---|---:|---|
| 1 | `sparse_attn` | [`baseline_adapter.py`](sparse_attn/ascendc/baseline_adapter.py)（baseline） | base `12.773625 ms`；adapter `12.772110 ms` | [baseline report](sparse_attn/evidence/baseline_report.md) · [最终失败报告](sparse_attn/evidence/candidate_failure.md) · [verdict](sparse_attn/evidence/candidate_failure_verdict.json) |
| 2 | `index_topk` | [`candidate_007.py`](index_topk/ascendc/candidate_007.py)（eager canonical） | `8.784070 → 8.298150 ms`，`1.059x` | [详细结果与采纳依据](index_topk/evidence.md) |
| 3 | `sinkhorn_normalize` | [`candidate_001.py`](sinkhorn_normalize/ascendc/candidate_001.py) + [`sinkhorn_normalize.cpp`](sinkhorn_normalize/ascendc/sinkhorn_normalize.cpp)（原生 Ascend C） | `1.524825 → 0.484020 ms`，`3.150345x` | [accepted report](sinkhorn_normalize/evidence/accepted_report.md) · [verdict](sinkhorn_normalize/evidence/accepted_verdict.json) |

参考实现：

- [`sparse_attn/base.py`](sparse_attn/base.py)
- [`index_topk/base.py`](index_topk/base.py)
- [`sinkhorn_normalize/base.py`](sinkhorn_normalize/base.py)

完整的测量口径、实现机制、采纳边界、跨算子经验、归档溯源与未声明结论见 [`summary.md`](summary.md)。

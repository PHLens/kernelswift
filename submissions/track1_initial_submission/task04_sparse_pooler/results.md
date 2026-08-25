# 赛题4：sparse_pooler 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | 1.60x | `mlu__triton_sparse_pooler_004.py` |
| `s60` | 0.79x | `s60__triton_sparse_pooler_001.py` |
| `maca` | — | `generic__triton_sparse_pooler_001.py` |
| `bi150` | 1.22x | `bi150__triton_sparse_pooler_001.py` |
| `ascend` | 1.51x | `ascend__triton_sparse_pooler_001.py` |

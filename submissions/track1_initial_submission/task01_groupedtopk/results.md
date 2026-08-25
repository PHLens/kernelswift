# 赛题1：groupedtopk 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | 6.56x | `mlu__triton_grouped_topk_004.py` |
| `s60` | 1.68x | `s60__triton_grouped_topk_003.py` |
| `maca` | 3.29x | `maca__triton_grouped_topk_001.py` |
| `bi150` | 1.71x | `bi150__triton_grouped_topk_009.py` |
| `ascend` | 2.84x | `ascend__triton_grouped_topk_002.py` |

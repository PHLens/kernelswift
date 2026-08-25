# 赛题2：flexattention 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | 7.08x | `mlu__triton_flexattention_003.py` |
| `s60` | 0.42x | `s60__triton_flexattention_001.py` |
| `maca` | — | `generic__triton_flexattention_001.py` |
| `bi150` | 0.61x | `bi150__triton_flexattention_001.py` |
| `ascend` | 1.45x | `ascend__triton_flexattention_002.py` |

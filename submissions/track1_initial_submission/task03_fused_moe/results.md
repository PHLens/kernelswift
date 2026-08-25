# 赛题3：fused_moe 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | 50.4x | `mlu__triton_fused_moe_005.py` |
| `s60` | 13.1x | `s60__triton_fused_moe_002.py` |
| `maca` | — | `generic__triton_fused_moe_002.py` |
| `bi150` | 6.60x | `bi150__triton_fused_moe_002.py` |
| `ascend` | 19.4x | `ascend__triton_fused_moe_003.py` |

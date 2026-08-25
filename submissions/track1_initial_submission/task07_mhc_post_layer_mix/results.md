# 赛题7：mhc_post_layer_mix 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | — | `generic__triton_mhc_post_layer_mix_001.py` |
| `s60` | 0.56x | `s60__triton_mhc_post_layer_mix_001.py` |
| `maca` | 31.66x | `maca__triton_mhc_001.py` |
| `bi150` | 1.20x | `bi150__triton_mhc_post_layer_mix_001.py` |
| `ascend` | 3.64x | `ascend__candidate_001.py` |

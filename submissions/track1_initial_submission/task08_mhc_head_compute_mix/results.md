# 赛题8：mhc_head_compute_mix 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | — | `generic__triton_mhc_head_compute_mix_001.py` |
| `s60` | 6.8x | `s60__triton_mhc_head_compute_mix_001.py` |
| `maca` | 14.07x | `maca__triton_mhcc_001.py` |
| `bi150` | 7.79x | `bi150__triton_mhc_head_compute_mix_001.py` |
| `ascend` | 9.00x | `ascend__candidate_001.py` |

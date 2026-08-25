# 赛题10：mhc_head_compute_mix_backward 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | — | `generic__triton_mhc_head_compute_mix_backward_001.py` |
| `s60` | 1.26x | `s60__triton_mhc_head_compute_mix_backward_001.py` |
| `maca` | — | `generic__triton_mhc_head_compute_mix_backward_001.py` |
| `bi150` | 1.76x | `bi150__triton_mhc_head_compute_mix_backward_001.py` |
| `ascend` | 1.03x | `ascend__triton_mhc_mix_bwd_001.py` |

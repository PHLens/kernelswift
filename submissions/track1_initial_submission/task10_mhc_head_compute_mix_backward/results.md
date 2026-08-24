# 赛题10：mhc_head_compute_mix_backward 性能测试结果

当前整理自仓库中已验证的 campaign / final summary / outcome 记录，作为本初版提交附带的性能结果。

| Backend | Current result | Selected submission file |
|---|---|---|
| `mlu` | — | `通用 Triton fallback` |
| `s60` | 🟡 1.26x | `triton_mhc_head_compute_mix_backward_001.py` |
| `maca` | — | `通用 Triton fallback` |
| `bi150` | ✅ 1.76x | `triton_mhc_head_compute_mix_backward_001.py` |
| `ascend` | 🟡 1.03x | `triton_mhc_mix_bwd_001.py` |

## 说明

- 本文件记录的是当前已验证结果与本次提交选择的实现文件对应关系。
- 对于没有专项优化版本的后端，`submission.py` 会退回到该赛题的通用 Triton fallback。
- 统一入口 `submission.py` 会按照后端分发到相应实现文件。

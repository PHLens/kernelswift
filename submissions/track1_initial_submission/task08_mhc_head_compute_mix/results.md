# 赛题8：mhc_head_compute_mix 性能测试结果

当前整理自仓库中已验证的 campaign / final summary / outcome 记录，作为本初版提交附带的性能结果。

| Backend | Current result | Selected submission file |
|---|---|---|
| `mlu` | — | `通用 Triton fallback` |
| `s60` | ✅ 6.8x | `triton_mhc_head_compute_mix_001.py` |
| `maca` | ✅ 14.07x | `triton_mhcc_001.py` |
| `bi150` | ✅ 7.79x | `triton_mhc_head_compute_mix_001.py` |
| `ascend` | ✅ 9.00x | `candidate_001.py` |

## 说明

- 本文件记录的是当前已验证结果与本次提交选择的实现文件对应关系。
- 对于没有专项优化版本的后端，`submission.py` 会退回到该赛题的通用 Triton fallback。
- 统一入口 `submission.py` 会按照后端分发到相应实现文件。

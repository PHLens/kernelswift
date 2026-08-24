# 赛题9：centre_random_augmentation 性能测试结果

当前整理自仓库中已验证的 campaign / final summary / outcome 记录，作为本初版提交附带的性能结果。

| Backend | Current result | Selected submission file |
|---|---|---|
| `mlu` | — | `通用 Triton fallback` |
| `s60` | 🟡 0.95x | `triton_centre_random_augmentation_001.py` |
| `maca` | — | `通用 Triton fallback` |
| `bi150` | ✅ 4.49x | `triton_centre_random_augmentation_002.py` |
| `ascend` | ✅ 1.22x | `triton_centre_random_aug_001.py` |

## 说明

- 本文件记录的是当前已验证结果与本次提交选择的实现文件对应关系。
- 对于没有专项优化版本的后端，`submission.py` 会退回到该赛题的通用 Triton fallback。
- 统一入口 `submission.py` 会按照后端分发到相应实现文件。

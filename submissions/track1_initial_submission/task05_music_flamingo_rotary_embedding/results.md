# 赛题5：music_flamingo_rotary_embedding 性能测试结果

当前整理自仓库中已验证的 campaign / final summary / outcome 记录，作为本初版提交附带的性能结果。

| Backend | Current result | Selected submission file |
|---|---|---|
| `mlu` | — | `未纳入本初版提交` |
| `s60` | 🟡 0.90x | `triton_rotary_002.py` |
| `maca` | ✅ 2.38x | `triton_rotary_001.py` |
| `bi150` | ✅ 1.95x | `triton_music_flamingo_rotary_embedding_001.py` |
| `ascend` | ✅ 1.86x | `triton_rotary_001.py` |

## 说明

- 本文件记录的是当前已验证结果与本次提交选择的实现文件对应关系。
- 对于尚未纳入本初版提交的后端，表中记为 `—`。
- 统一入口 `submission.py` 会按照后端分发到相应实现文件。

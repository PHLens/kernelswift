# 赛题6：mm_encoder_attention 性能测试结果

当前整理自仓库中已验证的 campaign / final summary / outcome 记录，作为本初版提交附带的性能结果。

| Backend | Current result | Selected submission file |
|---|---|---|
| `mlu` | — | `未纳入本初版提交` |
| `s60` | 🟡 0.27x | `triton_mm_encoder_attention_001.py` |
| `maca` | 🟡 0.91x | `triton_mha_002.py` |
| `bi150` | 🟡 0.55x | `triton_mm_encoder_attention_001.py` |
| `ascend` | 🟡 0.92x | `triton_attn_001.py` |

## 说明

- 本文件记录的是当前已验证结果与本次提交选择的实现文件对应关系。
- 对于尚未纳入本初版提交的后端，表中记为 `—`。
- 统一入口 `submission.py` 会按照后端分发到相应实现文件。

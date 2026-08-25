# 赛题5：music_flamingo_rotary_embedding 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | — | `generic__triton_music_flamingo_rotary_embedding_001.py` |
| `s60` | 0.90x | `s60__triton_rotary_002.py` |
| `maca` | 2.38x | `maca__triton_rotary_001.py` |
| `bi150` | 1.95x | `bi150__triton_music_flamingo_rotary_embedding_001.py` |
| `ascend` | 1.86x | `ascend__triton_rotary_001.py` |

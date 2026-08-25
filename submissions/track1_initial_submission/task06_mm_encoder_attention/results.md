# 赛题6：mm_encoder_attention 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | — | `generic__triton_mm_encoder_attention_001.py` |
| `s60` | 0.27x | `s60__triton_mm_encoder_attention_001.py` |
| `maca` | 0.91x | `maca__triton_mha_002.py` |
| `bi150` | 0.55x | `bi150__triton_mm_encoder_attention_001.py` |
| `ascend` | 0.92x | `ascend__triton_attn_001.py` |

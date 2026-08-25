# 赛题9：centre_random_augmentation 性能测试结果

| Backend | Speedup | Implementation |
|---|---|---|
| `mlu` | — | `generic__triton_centre_random_augmentation_001.py` |
| `s60` | 0.95x | `s60__triton_centre_random_augmentation_001.py` |
| `maca` | — | `generic__triton_centre_random_augmentation_001.py` |
| `bi150` | 4.49x | `bi150__triton_centre_random_augmentation_002.py` |
| `ascend` | 1.22x | `ascend__triton_centre_random_aug_001.py` |

# Track 1 Performance Summary

| Task | Operator | Current backend results |
|---|---|---|
| 01 | `groupedtopk` | mlu 6.56x · s60 1.68x · maca 3.29x · bi150 1.71x · ascend 2.84x |
| 02 | `flexattention` | mlu 7.08x · s60 0.42x · bi150 0.61x · ascend 1.45x |
| 03 | `fused_moe` | mlu 50.4x · s60 13.1x · bi150 6.60x · ascend 19.4x |
| 04 | `sparse_pooler` | mlu 1.60x · s60 0.79x · bi150 1.22x · ascend 1.51x |
| 05 | `music_flamingo_rotary_embedding` | s60 0.90x · maca 2.38x · bi150 1.95x · ascend 1.86x |
| 06 | `mm_encoder_attention` | s60 0.27x · maca 0.91x · bi150 0.55x · ascend 0.92x |
| 07 | `mhc_post_layer_mix` | s60 0.56x · maca 31.66x · bi150 1.20x · ascend 3.64x |
| 08 | `mhc_head_compute_mix` | s60 6.8x · maca 14.07x · bi150 7.79x · ascend 9.00x |
| 09 | `centre_random_augmentation` | s60 0.95x · bi150 4.49x · ascend 1.22x |
| 10 | `mhc_head_compute_mix_backward` | s60 1.26x · bi150 1.76x · ascend 1.03x |

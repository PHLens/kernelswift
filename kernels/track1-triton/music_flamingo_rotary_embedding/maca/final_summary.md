# Final Summary — MusicFlamingoRotaryEmbedding C500 (MACA)

- schema_version: 1
- skill_version: 2.0.0
- run_epoch: 1
- run_branch: kernel-opt/music-flamingo-rotary-c500-20260818
- base_branch: dev
- base_commit: e8533192f65ed4610a4b59859f1969ea83955f87
- measurement_fingerprint: 486242286573efe11bdd7b852247cb0ed4d63113e0e41c7c432ab65e654a6518
- stop_reason: user-intervention
- stop_timestamp: 2026-08-18T20:35:00Z
- total_rounds: 2
- accepted_round: 001
- canonical_kernel: triton_rotary_001.py

## Outcome

- baseline wall: 0.190557 ms (warmup 50 / repeat 100)
- final wall: 0.080036 ms
- improvement: +55.57% (device 50.95 -> 16.90 us/call, kernels/call 11 -> 1)

## Round Summary

| Round | Decision | Result | Wall ms | Improvement | Canonical |
|---:|---|---|---:|---:|---|
| 000 | Phase 0 | baseline | 0.190557 | - | baseline_adapter.py |
| 001 | H-001 kernel-fusion | accepted | 0.080036 | +55.57% | triton_rotary_001.py |
| 002 | abort (host-bound) | aborted | - | - | triton_rotary_001.py |

## Why stopped

Round 001 fused the whole rotary-embedding chain (batch/time broadcast,
concatenate, angle-scale, cos, sin — 11 PyTorch elementwise kernels) into one
direct-launch Triton-MACA elementwise kernel over `(B*SEQ, 2*dim) = (128, 128)`
output elements. This collapsed 11 launches -> 1 and removed all intermediate
materializations, yielding +55.57% wall.

Round 002 Designer performed Level 2 decomposition of the post-fusion
80.036 us/call wall and found the remaining ~63 us/call is harness-fixed:

- device `_rotary_embed_fused_kernel` = 16.90 us/call (21.1% of wall)
- harness `time_forward` times `set_seed(seed)` + `model.forward()` +
  `sync_devices()` (full `torch.cuda.synchronize()`) per sample; both
  `set_seed` and `sync_devices()` are harness-owned and un-optimizable.

A 5% gain = 4.0 us/call. All candidate-owned levers are falsified or sub-threshold:
1. output buffer coalescing — groupedtopk archive falsified this family at -13.71%
2. single-kernel internal tuning — device is not the binding constraint at
   device_ratio 0.211; shrinking 16.9 us device does not move the ~63 us host tail
3. num_warps/grid — num_warps=1 is the only Constrained-safe value (warp_size=64)

Conclusion: host-bound / measurement-bound. The accepted `triton_rotary_001.py`
(+55.57%) is the correct canonical.

## Resume constraints

Resume only with a new candidate-owned mechanism backed by same-runtime
evidence (>= 4.0 us/call compressible candidate-owned cost), or a user-mandated
measurement-regime change (new run epoch + fingerprint).

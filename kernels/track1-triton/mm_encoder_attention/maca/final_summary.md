# Final Summary — MMEncoderAttention C500 (MACA)

- schema_version: 1
- skill_version: 2.0.0
- run_epoch: 2
- run_branch: kernel-opt/mm-encoder-attention-c500-20260818
- base_branch: dev
- base_commit: 99cd9f4ee002f83e21c7c639c891ebcc2d5ba689
- measurement_fingerprint: 29ecde127206fc1808c2d7f28951e44ee55a257aadfda78517e64d3493ce1862
- stop_reason: user-intervention
- stop_timestamp: 2026-08-19T02:10:00Z
- total_rounds: 2
- accepted_round: 002
- canonical_kernel: triton_mha_002.py

## Outcome

- baseline wall: 0.115761 ms (flash attention SDPA)
- final wall: 0.127777 ms (hand-written Triton MHA)
- improvement vs baseline: -15.26% (slower than flash attention, expected)
- improvement vs round-001 canonical: +23.54% (transpose-copy elimination)

## Round Summary

| Round | Decision | Result | Wall ms | Improvement | Canonical |
|---:|---|---|---:|---:|---|
| 000 | Phase 0 | baseline | 0.115761 | - | baseline_adapter.py |
| 001 | H-001 fused-mha-kernel | accepted | 0.164166 | -48.16% | triton_mha_001.py |
| 002 | H-002 remove-transpose-copy | accepted | 0.127777 | +23.54% | triton_mha_002.py |

## Purpose of epoch 2

Epoch 1 concluded "measurement-bound abort" because SDPA already lowers to flash
attention. Epoch 2 re-opened with a new intent: the competition requires every
operator to ship a Triton kernel. The deliverable is a hand-written Triton MHA
kernel (`triton_mha_002.py`), correct (allclose 1e-2) and reasonably-optimized,
even though it does not beat the hardware-optimized flash attention.

## Why stopped

The hand-written Triton MHA kernel is the required deliverable and has been
optimized as far as the C500 profile allows:
- Round 001 produced a correct fused MHA kernel (manual tl.sum dot, two-pass
  max-subtracted softmax, fp32 accumulate).
- Round 002 removed the four `.contiguous()` transpose-copy kernels by loading
  q/k/v directly from the original contiguous layout, collapsing 5 kernels -> 2
  (1 fused `_mha_fwd_kernel` + 1 unavoidable output reshape), improving wall
  +23.54% over round 001.

The remaining dominant cost (`_mha_fwd_kernel` ~64.85 us/call) is the manual
`tl.sum` dot over head_size=64, which cannot be accelerated without `tl.dot`
(Unknown on C500). No further candidate-owned lever has a defensible >=5% wall
path. This is measurement-bound with respect to the flash-attention floor.

## Resume constraints

Resume only if `tl.dot` becomes available on triton_maca (would enable a true
matrix-multiply attention), or with a same-runtime microbenchmark proving a
compressible candidate-owned bottleneck.

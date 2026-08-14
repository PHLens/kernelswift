# Designer State

Concise evidence considered, rejected hypotheses, and open semantic questions.
Historical candidates are labeled noncanonical. No runtime claims without a
Verifier report.

## Round 001

- Selected bottleneck: mixed (device_ratio 0.1979, host-bound/mixed boundary).
  Six device kernels (relu 13.65 + log1p 26.17 + 4x max-pool 29.62 = 158.30
  us/call, 87.9% of device time) are the fusion target. The 4 max-pool kernels
  are launched from a Python for-loop over seq_lens.tolist() (D2H sync + 4
  sequential dispatches).
- Intervention: fuse relu+log1p+per-segment max into one Triton kernel, one
  launch per forward. Eliminates the Python loop and D2H sync; replaces 6 device
  kernels with 1.
- change_scope: mixed (kernel dataflow + host dispatch path). Both pieces are
  inseparable (the fused kernel is what removes the loop) and separately
  observable (kernel_count_per_call, device_us_per_call, host_sync_count).
- Expected wall improvement: 15%. Adoption threshold: 5%.
- Open primitive questions for Coder: tl.maximum, tl.where, tl.log are not
  explicitly listed in the triton_mlu Supported table. Coder must run a local
  compile-and-run probe. Fallback if tl.maximum is unavailable: tl.where-based
  comparison; if that also fails: tl.argmax (explicitly Supported) + indexed
  load.
- Rejected alternatives for this round:
  - Fuse only relu+log1p (elementwise): saves ~40 us device + 1 launch, unlikely
    to clear 5% wall threshold alone.
  - Fuse the decoder matmul (MLUFusedMatMulGepm 89.42 us/call) into Triton:
    larger change boundary, requires tl.dot with [83,768]x[768,30522], higher
    risk. Deferred to a future round if round 001 succeeds and the decoder
    matmul becomes the new dominant bottleneck.
  - Host-only vectorized segment max via torch ops: would still launch per-chunk
    kernels or require scatter-reduction, not a clean single-launch solution.
- Noncanonical history: none (round 001 is the first optimization round).

## Round 002

- Selected bottleneck: device-bound (candidate device_ratio 0.346, mixed class;
  the fused `_sparse_pooler_max_kernel` is the dominant device kernel at 98.73
  us/call, 47.0% of candidate device time, and is 30.86 us/call SLOWER than the
  6 library kernels it replaced at 67.87 us/call combined).
- Intervention: tune the fused `_sparse_pooler_max_kernel` by increasing BLOCK_V
  from 1024 to 2048, halving the vocab tile count from 30 to 15 and the total
  grid programs from 120 to 60. This attacks per-program overhead (prefix scan
  setup, loop control, program dispatch) which is the likely cause of the
  regression given the fused kernel has ~6x less memory traffic than the library
  kernels yet is slower.
- change_scope: kernel. The change is a single kernel constexpr (BLOCK_V) with a
  trivially implied host-side grid recomputation. No allocation reuse, output
  caching, lifecycle, or concurrency changes. Host Plan is not-applicable.
- Expected wall improvement: 7.0%. Adoption threshold: 5%. Justification:
  recovering the 30.86 us/call device regression alone yields 5.09% wall
  improvement (30.86/606.76); halving the grid from 120 to 60 programs adds
  further host launch overhead savings of an estimated 10-20 us/call
  (1.7-3.3% wall).
- Open primitive questions for Coder:
  - BLOCK_V=2048 register pressure: the `acc` and `vocab_tile` tiles each hold
    BLOCK_V fp32 values (~8 KB at 1024, ~16 KB at 2048). Coder must verify the
    kernel compiles and produces correct output at BLOCK_V=2048 on MLU590-H8.
    Fallback: BLOCK_V=1536 or remain at 1024 if compile fails or registers spill.
  - BLOCK_V=4096 is a secondary fallback probe (~32 KB register pressure) and
    may fail to compile; use only if 2048 does not clear the adoption threshold.
  - num_warps tuning: num_warps=1 is proven and normative. num_warps=2 is known
    to fail and must not be used. Other values are Unknown; Coder may probe
    locally but must fall back to num_warps=1 on any compile/correctness failure.
  - The mask `v_mask = v_offs < vocab_size` must be preserved on both tl.load
    (other=-inf) and tl.store. With BLOCK_V=2048, the last vocab tile is partial
    (30522 - 14*2048 = 1850 elements).
- Rejected alternatives for this round:
  - (B) Fuse the decoder matmul (MLUFusedMatMulGepm, 90.36 us/call) into Triton
    using tl.dot with shape [83,768]x[768,30522]: larger change boundary, higher
    risk, and the matmul is already a fused MLU library kernel. Deferred to a
    future round if the fused max kernel is tuned and the decoder matmul remains
    the dominant bottleneck.
  - (C) Host-side launcher reduction / allocation reuse: targets the remaining
    ~396 us/call host time but requires Host Plan lifecycle changes (output
    buffer caching, fast_libentry). Deferred to a future round; the device-side
    regression in the fused kernel is the proven headroom and lower-risk target.
  - Reduction strategy that loads each logits row once and accumulates across
    vocab tiles: would require cross-program synchronization or a two-pass
    kernel, increasing complexity and change boundary. The BLOCK_V tiling change
    is a simpler, lower-risk first step.
- Noncanonical history: the Round 001 fused kernel (triton_sparse_pooler_001.py)
  is the canonical starting point. No rejected candidates exist for Round 002.

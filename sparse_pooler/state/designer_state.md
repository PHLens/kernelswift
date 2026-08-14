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

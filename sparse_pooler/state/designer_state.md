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

## Round 003

- Selected bottleneck: device-bound (candidate device_ratio 0.346, mixed class).
  The decoder matmul (MLUFusedMatMulGepm, 90.36 us/call, 43.0% of device time)
  and the existing fused _sparse_pooler_max_kernel (98.73 us/call, 47.0%) together
  account for 189.09 us/call — 90.0% of candidate device time. The decoder matmul
  materializes the intermediate logits tensor [83, 30522] fp32 (10.16 MB) in global
  memory, which the fused reduction kernel re-reads.
- Intervention: fuse the decoder matmul (via tl.dot with K-dimension tiling), bias
  addition, relu, log1p, and per-segment max reduction into a single Triton kernel
  launched once per forward. Eliminates the library MLUFusedMatMulGepm decoder
  matmul kernel and the existing fused _sparse_pooler_max_kernel; avoids
  materializing the intermediate logits tensor in global memory.
- change_family: kernel-matmul-fusion (different from Round 001 kernel-fusion and
  Round 002 kernel-tile-tuning, as required by the v2 contract after a
  no-improvement result).
- change_scope: mixed (kernel dataflow + host dispatch path). The decoder matmul
  is fused into the Triton kernel (device-side), and the host dispatch path
  removes the self.decoder(...) library op call. Both pieces are inseparable (the
  fused kernel is what removes the library op) and separately observable
  (decoder_matmul_kernel_count_per_call, total_kernel_count_per_call,
  device_us_per_call, fused_kernel_us_per_call).
- Expected wall improvement: 8.0%. Adoption threshold: 5%. Justification: the two
  replaced kernels cost 189.09 us/call combined; the new fused kernel avoids the
  10.16 MB intermediate tensor traffic and merges two kernel launches into one.
  Conservative estimate: new fused kernel ~130 us/call, saving ~59 us device +
  ~6 us host = ~65 us / 607 us = ~10.7% wall. The 8.0% expectation is conservative.
- Open primitive questions for Coder:
  - tl.dot is Supported per triton_mlu target profile but must be probed locally
    with fp32 inputs of shape [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_V] where
    BLOCK_M=32 (max seq_len=25, padded) and BLOCK_K tiles hidden_size=768. If
    tl.dot does not support these shapes/dtypes, Coder must report capability-miss
    (NOT fall back to the library op — that would be major-deviation).
  - K-dimension tiling (BLOCK_K in {64, 128, 256}) is required because the full
    weight tile [BLOCK_V, 768] is too large for registers. Coder must accumulate
    the tl.dot result across K tiles in a [BLOCK_M, BLOCK_V] accumulator.
  - BLOCK_M=32 (next power of 2 >= max(seq_len)=25) with row masking for rows
    >= seq_len in the tl.dot and max reduction.
  - BLOCK_V starting point: 256 or 512 (smaller than the 1024 used in the
    existing fused kernel, due to additional register pressure from the tl.dot
    accumulator and weight tiles). Optional probe space: BLOCK_V in {256, 512,
    1024}, BLOCK_K in {64, 128, 256}, subject to compile and correctness.
  - num_warps=1 is normative and proven. num_warps=2 must not be used. Other
    values are Unknown; Coder may probe locally but must fall back to num_warps=1.
  - Decoder weight layout: nn.Linear stores weight as [vocab_size, hidden_size]
    = [30522, 768]. The kernel needs weight_tile = decoder_weight[v_start:v_start
    +BLOCK_V, k:k+BLOCK_K] loaded as [BLOCK_V, BLOCK_K], used transposed in
    tl.dot(hidden_tile, weight_tile.T) to compute [BLOCK_M, BLOCK_V]. Coder must
    handle stride layout correctly; pre-transposing at init risks breaking
    load_state_dict (harness updates self.decoder.weight but not a pre-transposed
    copy).
  - Decoder bias [30522] loaded as 1-D tile [BLOCK_V] and broadcast across
    [BLOCK_M, BLOCK_V] logits tile.
- Rejected alternatives for this round:
  - (A) kernel-reduction-strategy: a different reduction strategy for the existing
    fused kernel (e.g., load each logits row once and accumulate across vocab
    tiles, or a two-pass kernel). This is a kernel-scope change but the fused
    kernel is 98.73 us/call and the decoder matmul is 90.36 us/call — the
    reduction strategy alone cannot recover enough device time to clear 5% wall
    (the 30.86 us/call device regression in the fused kernel is only 5.09% of
    wall, and Round 002 showed the kernel is not easily tuned). The decoder
    matmul fusion is a larger opportunity (189.09 us/call combined).
  - (C) host-allocation-reuse: targets the remaining ~396 us/call host time but
    requires Host Plan lifecycle changes (output buffer caching, fast_libentry).
    Independent of device-side work and deferred to a future round. The
    device-side decoder matmul fusion is the larger proven headroom.
  - Dense matmul + GELU + LayerNorm fusion: the three remaining library ops
    (8.42 + 7.21 + 5.40 = 21.03 us/call) are too small to clear 5% wall alone
    (21.03/607 = 3.5%). Deferred to a future round if the decoder matmul fusion
    succeeds and these become the new bottleneck.
- Noncanonical history: the Round 002 BLOCK_V=2048 tuning (triton_sparse_pooler_002.py)
  is a rejected candidate and never a starting point. The Round 001 fused kernel
  (triton_sparse_pooler_001.py) remains the canonical starting point. Round 002
  evidence (BLOCK_V=1024 is best-known, kernel-tile-tuning is exhausted) is
  noncanonical for future rounds but informs the choice of change_family.

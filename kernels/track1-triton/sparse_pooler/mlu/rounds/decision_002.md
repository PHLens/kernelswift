# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_sparse_pooler_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"tune the fused _sparse_pooler_max_kernel by increasing BLOCK_V from 1024 to 2048, halving the vocab tile count from 30 to 15 and the total grid programs from 120 to 60, to recover the 30.86 us/call device regression against the six library kernels it replaced and reduce per-program launch overhead","allowed_changes":["BLOCK_V constexpr value in _sparse_pooler_max_kernel changed from 1024 to 2048","corresponding host-side grid computation update: num_vocab_tiles = cdiv(vocab_size, BLOCK_V) recomputed with the new BLOCK_V","optional local probe of BLOCK_V=4096 as a fallback if BLOCK_V=2048 does not clear the adoption threshold, subject to compile and correctness verification","optional local probe of num_warps values other than 1 (proven) and 2 (known to fail), subject to compile and correctness verification, with num_warps=1 as the proven fallback"],"invariants":["ModelNew public constructor and forward signature unchanged","output is a Python list of num_seq tensors each of shape [vocab_size] dtype fp32 device mlu:0","numerical semantics: log(1+relu(decoder_logits)) pooled per sequence with max reduction within atol=1e-2 rtol=1e-2 equal_nan=True","caller-selected device and current stream preserved","dense GELU LayerNorm decoder matmul pipeline remains PyTorch library ops unchanged","kernel count per call remains 5 (no kernels added or removed; only the fused kernel's tiling parameter changes)","load_state_dict compatibility maintained","num_warps=2 must not be used (known to fail on this runtime per triton_mlu target profile)"],"expected_wall_improvement_pct":7.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor logits shape=[total_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tensor seq_lens shape=[num_seq] dtype=int32 layout=contiguous memory=global
tensor out shape=[num_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tile vocab_tile shape=[BLOCK_V] dtype=fp32 memory=register
tile acc shape=[BLOCK_V] dtype=fp32 memory=register
scalar pid_s dtype=int32 memory=register
scalar pid_v dtype=int32 memory=register
scalar seq_offset dtype=int32 memory=register
scalar seq_len dtype=int32 memory=register

# O Operations
load pid_s <- program_id(0)
load pid_v <- program_id(1)
load seq_len <- seq_lens[pid_s]
compute seq_offset <- sum(seq_lens[0:pid_s])
alloc acc <- full([BLOCK_V], -inf, dtype=fp32)
load vocab_tile <- logits[seq_offset + row, pid_v * BLOCK_V : pid_v * BLOCK_V + BLOCK_V]
compute vocab_tile <- log(1 + relu(vocab_tile))
compute acc <- maximum(acc, vocab_tile)
store out[pid_s, pid_v * BLOCK_V : pid_v * BLOCK_V + BLOCK_V] <- acc

# C Control
parallel pid_s over num_seq
parallel pid_v over cdiv(vocab_size, BLOCK_V)
guard pid_s < num_seq
guard pid_v * BLOCK_V < vocab_size
for row in 0:seq_len
guard row < seq_len
end

# H Target Hints
target=triton_mlu
num_warps=1
BLOCK_V=2048
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only tuning of BLOCK_V in the fused _sparse_pooler_max_kernel; the host-side grid computation change is a direct consequence of the kernel constexpr and introduces no allocation reuse, output caching, lifecycle, or concurrency changes"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"tune the fused _sparse_pooler_max_kernel by increasing BLOCK_V from 1024 to 2048, halving the vocab tile count from 30 to 15 and the total grid programs from 120 to 60, to recover the 30.86 us/call device regression against the six library kernels it replaced and reduce per-program launch overhead","expected_causal_chain":["BLOCK_V increases from 1024 to 2048 so the vocab dimension is covered by 15 tiles instead of 30","the fused kernel grid drops from (4,30)=120 programs to (4,15)=60 programs","per-program overhead (prefix scan, loop setup, program dispatch) is halved because half as many programs execute","the fused kernel device time decreases from 98.73 us/call toward or below the 67.87 us/call combined cost of the six library kernels it replaced","total device_us_per_call decreases from 210.12 us/call","wall time decreases by at least 5%"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"fused_kernel_us_per_call","expectation":"decrease from 98.73 us/call; target is to recover the 30.86 us/call regression and approach or fall below the 67.87 us/call combined cost of the six replaced library kernels"},{"name":"device_us_per_call","expectation":"decrease from 210.12 us/call as the fused kernel cost drops; the four non-fused kernels (decoder matmul 90.36, dense matmul 8.42, LayerNorm 7.21, GELU 5.40 us/call) are unchanged"},{"name":"fused_kernel_grid_programs","expectation":"decrease from 120 (4 sequences x 30 vocab tiles) to 60 (4 sequences x 15 vocab tiles)"}],"guardrails":["correctness:pass","output is a Python list of num_seq tensors each [vocab_size] fp32 mlu:0","numerical semantics log(1+relu(decoder_logits)) max-pooled per sequence preserved within atol=1e-2 rtol=1e-2 equal_nan=True","caller-selected device and current stream preserved","dense GELU LayerNorm decoder matmul pipeline unchanged","ModelNew public constructor and forward signature unchanged","load_state_dict compatibility maintained","kernel_count_per_call remains 5 (no kernels added or removed)","num_warps=2 is not used (known to fail on this runtime)"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No entry matches this intervention.
  The recorded failures concern winner-tree expert selection, sort networks,
  dynamic `tl.gather` compaction, and cumsum compaction in grouped top-k — none
  apply to a BLOCK_V tiling parameter change in a fused elementwise-plus-segment-max
  reduction kernel.
- Consulted `references/bottleneck-judgment.md`. The candidate device ratio is 0.346
  (mixed). The fused `_sparse_pooler_max_kernel` dominates device time at 98.73 us/call
  (47.0% of candidate device time) and is slower than the 6 library kernels it replaced
  (67.87 us/call combined). The intervention targets this single dominant kernel's
  tiling parameter — a separately observable device mechanism. The non-fused kernels
  (decoder matmul, dense matmul, LayerNorm, GELU) are outside the change boundary and
  unchanged.
- Consulted `prompts/coder_targets/triton_mlu.md`. `num_warps=1` is proven and is the
  primary launch configuration. `num_warps=2` failed in the flexattention experiment and
  must not be used. Every other `num_warps` value is Unknown until probed locally. The
  decision uses `num_warps=1` as the normative value; Coder may probe other values
  locally but must fall back to `num_warps=1` if the probe fails to compile, produces
  incorrect output, or does not improve wall time.
- `BLOCK_V=2048` is a kernel constexpr change. The current `BLOCK_V=1024` compiled and
  ran correctly. A larger BLOCK_V increases register pressure: the `acc` tile and the
  `vocab_tile` each hold BLOCK_V fp32 values, so BLOCK_V=2048 requires approximately
  16 KB of register space for the two tiles combined (vs 8 KB at BLOCK_V=1024). Coder
  must verify that the kernel compiles and produces correct output at BLOCK_V=2048 on
  the MLU590-H8 architecture. If BLOCK_V=2048 causes a compile failure or register
  spill, the fallback is BLOCK_V=1536 or remaining at 1024 while probing other tuning
  axes. BLOCK_V=4096 is a secondary fallback probe (~32 KB register pressure) and may
  fail to compile.
- The vocabulary size 30522 is not a power of 2. With BLOCK_V=2048, the last vocab tile
  is partial: `30522 - 14 * 2048 = 30522 - 28672 = 1850` elements. The existing
  `v_mask = v_offs < vocab_size` mask handles this correctly. Coder must preserve the
  mask on both `tl.load` (with `other=-float("inf")`) and `tl.store`.
- The kernel's on-device prefix scan (`seq_offset = sum(seq_lens[0:pid_s])`) is
  unchanged. With `num_seq=4` this is at most 3 extra `tl.load` calls per program —
  negligible and unaffected by the BLOCK_V change.
- `tl.maximum`, `tl.where`, and `tl.log` are used in the kernel body and were proven
  to work in Round 001 (the candidate passed correctness and the fused kernel ran
  successfully). No new primitives are introduced in this round.
- The harness AST loader strips module-level non-literal assignments. The Triton kernel
  definition is at module level (decorated with `@triton.jit`) and was retained in
  Round 001. No changes to the module structure are needed.
- Per `references/invariants.md`, this change does not introduce buffer caching, output
  reuse, or any cross-forward state. The output tensor is still allocated per-forward
  with `torch.empty`. The Host Plan is not-applicable because the change is purely a
  kernel constexpr tuning with a trivially implied host-side grid recomputation.

## Rationale and Evidence

The accepted Round 001 report (`rounds/report_001.md`) records:

- Benchmark wall time median: 0.606758 ms (606.76 us/call), a 33.39% improvement over
  the baseline that was accepted in Round 001.
- Device time: 210.12 us/call; device_ratio: 0.346 (mixed).
- Kernel count: 5 per call (down from 10).
- The fused `_sparse_pooler_max_kernel` is the new dominant device kernel at
  98.73 us/call — 47.0% of candidate device time. It is SLOWER on the device than the
  6 library kernels it replaced (relu 13.67 + log1p 26.17 + 4x reduceKernelMaxIndex
  28.03 = 67.87 us/call combined), a 30.86 us/call device regression.
- The current kernel uses `BLOCK_V=1024`, producing a grid of `(4, 30) = 120` programs.
  Each program processes one sequence and one vocab tile of 1024 elements, looping
  over `seq_len` rows.

The 30.86 us/call device regression is the proven headroom. The fused kernel has
approximately 6x less memory traffic than the 6 library kernels (it reads the logits
tensor once and writes the output once, vs the library kernels' multiple read/write
passes for relu, log1p, and 4 max-pool reductions). Despite this, the fused kernel is
slower, which indicates the bottleneck is per-program overhead (prefix scan setup, loop
control, program dispatch) rather than memory bandwidth. Halving the number of programs
from 120 to 60 by increasing BLOCK_V to 2048 directly attacks this overhead.

The expected wall improvement of 7.0% is justified as follows:

1. Recovering the 30.86 us/call device regression (matching the 67.87 us/call combined
   library kernel cost) yields 30.86 us of device savings. At the current wall time of
   606.76 us/call, this alone is 5.09% wall improvement if fully captured.
2. Halving the grid from 120 to 60 programs reduces launch dispatch overhead on the
   host side. The host launch overhead for 120 vs 60 programs is expected to save an
   additional 10-20 us/call, contributing 1.7-3.3% wall improvement.
3. The combined expected improvement is 7-8%, and the 7.0% expectation is conservative.

The decoder matmul (`MLUFusedMatMulGepm`, 90.36 us/call) is the second-largest device
kernel and the largest non-fused kernel. Fusing it into Triton would require a `tl.dot`
matmul primitive with shape [83,768]x[768,30522] — a larger change boundary and higher
risk. That fusion is a candidate for a future round if this round succeeds and the
decoder matmul remains the dominant bottleneck after the fused kernel is tuned.

Host-side launcher reduction and allocation reuse (the remaining ~396 us/call host
time) require Host Plan lifecycle changes and are deferred to a future round. The
current round targets the single dominant device kernel with a low-risk tiling
parameter change.

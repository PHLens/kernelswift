# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"mixed"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse relu, log1p, and per-sequence max pooling into a single Triton kernel launched once per forward, eliminating the Python for-loop over seq_lens.tolist() and replacing six device kernels (relu, log1p, 4x reduceKernelMaxIndex) with one fused reduction kernel","allowed_changes":["new Triton kernel performing elementwise relu+log1p followed by per-segment max reduction over the vocabulary dimension","ModelNew.forward dispatch path: replace the Python for-loop and per-chunk torch.max calls with a single fused kernel launch","device-side offset computation from seq_lens tensor inside the kernel to avoid D2H synchronization"],"invariants":["ModelNew public constructor and forward signature","output is a Python list of num_seq tensors each of shape [vocab_size] dtype fp32 device mlu:0","numerical semantics: log(1+relu(decoder_logits)) pooled per sequence with max reduction","tolerance atol=1e-2 rtol=1e-2 equal_nan=True","caller-selected device and current stream preserved","dense, GELU, LayerNorm, and decoder matmul remain PyTorch library ops unchanged"],"expected_wall_improvement_pct":15.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor logits shape=[total_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tensor seq_lens shape=[num_seq] dtype=int32 layout=contiguous memory=global
tensor out shape=[num_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tile vocab_tile shape=[BLOCK_V] dtype=fp32 memory=register
tile acc shape=[BLOCK_V] dtype=fp32 memory=register
scalar pid dtype=int32 memory=register
scalar seq_offset dtype=int32 memory=register
scalar seq_len dtype=int32 memory=register

# O Operations
load pid <- program_id(0)
load seq_len <- seq_lens[pid]
compute seq_offset <- sum(seq_lens[0:pid])
alloc acc <- zeros([BLOCK_V], dtype=fp32)
load vocab_tile <- logits[seq_offset + row, v_start:v_end]
compute vocab_tile <- log1p(relu(vocab_tile))
compute acc <- maximum(acc, vocab_tile)
store out[pid, v_start:v_end] <- acc

# C Control
parallel pid over num_seq
parallel v_start over vocab_size stride BLOCK_V
guard v_start < vocab_size
guard pid < num_seq
for row in 0:seq_len
guard row < seq_len
end

# H Target Hints
target=triton_mlu
num_warps=1
```

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward dispatch path","Triton kernel launch","output tensor allocation"],"state_owner":"ModelNew instance","lifetime":"model lifetime; kernel compiled once and cached by Triton JIT","allocation_reuse":"output list of num_seq tensors allocated per forward call; no cross-forward buffer caching in this round","cache_key":["seq_lens shape","num_seq","vocab_size","dtype","device"],"invalidation":"no persistent cache in this round; kernel autotune cache managed by Triton runtime","concurrency":"one ModelNew instance is not shared across concurrent forwards; output tensors are per-call","device_stream_behavior":"caller-selected device and current stream are preserved; no explicit torch.mlu.device() context is introduced; seq_lens offsets are computed on-device inside the kernel to avoid D2H synchronization","unchanged_behavior":["returned Python list of num_seq tensors","each output tensor shape [vocab_size]","each output tensor dtype fp32","each output tensor device mlu:0","numerical semantics log(1+relu(logits)) max-pooled per sequence","dense GELU LayerNorm decoder matmul pipeline unchanged","load_state_dict compatibility"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse relu, log1p, and per-sequence max pooling into a single Triton kernel launched once per forward, eliminating the Python for-loop over seq_lens.tolist() and replacing six device kernels (relu, log1p, 4x reduceKernelMaxIndex) with one fused reduction kernel","expected_causal_chain":["the Python for-loop over seq_lens.tolist() and four per-chunk torch.max dispatches are replaced by one Triton kernel launch","host-side D2H synchronization from seq_lens.tolist() is eliminated because offsets are computed on-device","device kernel count per call drops from 10 to 5 (relu, log1p, and 4x reduceKernelMaxIndex removed; one fused kernel added)","host dispatch overhead decreases and device kernel count decreases","wall time decreases by at least 5%"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease from 10 to 5"},{"name":"device_us_per_call","expectation":"decrease by roughly the 158 us/call contributed by relu+log1p+4x max-pool minus the fused kernel cost"},{"name":"host_sync_count_per_call","expectation":"decrease because seq_lens.tolist() D2H sync is eliminated"}],"guardrails":["correctness:pass","output is a Python list of num_seq tensors each [vocab_size] fp32 mlu:0","numerical semantics log(1+relu(decoder_logits)) max-pooled per sequence preserved within atol=1e-2 rtol=1e-2","caller-selected device and current stream preserved","dense GELU LayerNorm decoder matmul pipeline unchanged","ModelNew public constructor and forward signature unchanged","load_state_dict compatibility maintained"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No entry matches this intervention: the
  recorded failures concern winner-tree expert selection, sort networks, dynamic
  `tl.gather` compaction, and cumsum compaction in grouped top-k — none apply to a
  straightforward elementwise-plus-segment-max-reduction fusion.
- Consulted `references/bottleneck-judgment.md`. The device ratio of 0.1979 sits at
  the host-bound/mixed boundary. The intervention targets both the host-side Python
  loop (compressible: launcher and dispatch overhead) and the device-side kernel
  count (compressible: fusion of relu, log1p, and 4 max-pool kernels). Both pieces
  are inseparable — the fused kernel is what eliminates the loop — and separately
  observable via kernel_count_per_call and device_us_per_call.
- `tl.maximum` and `tl.where` are not explicitly listed in the triton_mlu Supported
  table. They are fundamental elementwise primitives. Coder must verify via a local
  compile-and-run probe that the max-reduction lowering works on this MLU runtime.
  If `tl.maximum` is unavailable, the fallback is a comparison-based update using
  `tl.where(acc < vocab_tile, vocab_tile, acc)`. If both are unavailable, the
  fallback is `tl.argmax` (explicitly Supported) over the sequence axis followed by
  an indexed load to recover the max value — less efficient but semantically
  correct. Any fallback must preserve the Evaluation Contract observables and
  guardrails.
- `log1p` and `relu` inside the kernel are elementwise math ops (`tl.log`, `tl.max`
  or comparison). Coder should verify `tl.log` and `tl.exp` availability via the
  same local probe. The baseline uses `torch.log1p` which is numerically stable for
  small positive inputs; the Triton implementation should use `tl.log(1.0 + x)` or
  a `tl.where` guard for the relu output (which is non-negative, so `1+x >= 1` and
  log1p is stable).
- The kernel computes `seq_offset = sum(seq_lens[0:pid])` on-device. With num_seq=4
  this is at most 3 extra `tl.load` calls per program — negligible. This avoids the
  D2H synchronization that `seq_lens.tolist()` triggers in the baseline.
- The vocabulary dimension (30522) is tiled with BLOCK_V. Coder must choose BLOCK_V
  to divide or bounds-check against 30522. A mask is required on the last tile.
- The harness AST loader strips module-level non-literal assignments. The Triton
  kernel definition and `fast_libentry` initialization (if used) must be inside the
  class body or at a location the loader retains. Per the target profile, both
  `from triton.runtime import fast_libentry` and
  `from triton.runtime.fast_libentry import fast_libentry` work here; Coder should
  probe the actual loader behavior.
- Per `references/invariants.md`, removing `torch.mlu.device()` is valid only when
  the caller already owns device selection. The baseline does not use an explicit
  `torch.mlu.device()` context, so this decision does not introduce or remove one.
- Output tensors are allocated per-forward. No cross-forward buffer caching is
  introduced in this round, avoiding the concurrency and invalidation complexity
  documented in `references/invariants.md` Buffer/Device/Stream Lifecycle rules.

## Rationale and Evidence

The accepted baseline report (`rounds/report_000.md`) records:

- Benchmark wall time median: 0.909974 ms (909.97 us/call).
- Device time: 180.05 us/call; device_ratio: 0.1979 — at the host-bound/mixed
  boundary.
- 10 device kernels per call. Six of them are the fusion target:
  - `MLUBlockKernel3StagePipelineClipFast` (relu): 13.65 us/call
  - `MLUBlockKernel5StagePipelineLog1pFast` (log1p): 26.17 us/call
  - `reduceKernelMaxIndex` (per-sequence max pool): 29.62 us/call x 4 = 118.48 us/call
  - Combined device cost of these six: 158.30 us/call (87.9% of device time).
- The 4 max-pool kernels are launched from a Python `for` loop over
  `seq_lens.tolist()`, which triggers a D2H synchronization and serializes 4
  dispatches on the host side.
- Roughly 730 us/call (~80% of wall) is host-side: Python loop, per-launch
  overhead, and sequential dispatch.

The intervention fuses these six kernels into one Triton kernel and eliminates the
Python loop. The expected causal chain is direct: removing 5 device kernel launches
and 1 D2H sync reduces both host dispatch overhead and device kernel count. Even if
the fused kernel is no faster per-element than the library kernels, the launch
consolidation and host-loop elimination alone are expected to clear the 5% adoption
threshold. A 15% expected wall improvement is justified by the fact that 158 us/call
of device work plus the host-side loop overhead (a substantial fraction of the 730
us/call host budget) is the target; capturing even half of the combined host+device
savings exceeds 5%.

The decoder matmul (`MLUFusedMatMulGepm`, 89.42 us/call) is left as a PyTorch
library op because it is already a fused MLU kernel and fusing it into Triton would
require a matmul primitive (`tl.dot`) with shape [83,768]x[768,30522], which is a
larger change boundary and higher risk. That fusion is a candidate for a future
round if this round succeeds and the decoder matmul becomes the new dominant
bottleneck.

# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse the 14-elementwise forward chain (arange, repeat_interleave, broadcast, cat, mul, cos, sin) into a single Triton kernel that reads the precomputed inv_freq and position_angles buffers and timestamps and writes both cos and sin outputs directly","allowed_changes":["kernel dataflow","ModelNew.forward elementwise body"],"invariants":["ModelNew public constructor and forward signature","output structure: tuple (cos, sin)","output shape [4,32,128] fp32 each","register_buffer inv_freq and position_angles precomputation in __init__ unchanged","numerical semantics: freqs = cat(batch_freqs, time_freqs) * (-timestamps*2pi)","harness entry points ModelNew/get_inputs/get_init_inputs"],"expected_wall_improvement_pct":60.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor timestamps shape=[B,SEQ] dtype=fp32 layout=row_major memory=global
tensor inv_freq shape=[DIM2] dtype=fp32 layout=contiguous memory=global
tensor position_angles shape=[MAX_SEQ,DIM] dtype=fp32 layout=row_major memory=global
tensor cos_out shape=[B,SEQ,2DIM] dtype=fp32 layout=row_major memory=global
tensor sin_out shape=[B,SEQ,2DIM] dtype=fp32 layout=row_major memory=global
scalar max_seq_len dtype=int
scalar dim dtype=int
scalar two_pi dtype=fp32

# O Operations
compute idx = program_id
compute b = idx // (SEQ*2DIM)
compute t = (idx // 2DIM) % SEQ
compute c = idx % 2DIM
compute half = c // 2
compute is_time = c >= dim
load freq_time <- position_angles[t,c-dim]
load freq_batch_raw <- inv_freq[half]
compute freq_batch = freq_batch_raw * (b/max_seq_len)
compute freq = is_time ? freq_time : freq_batch
load ts <- timestamps[b,t]
compute angle = -ts * two_pi
compute value = freq * angle
compute cosv = cos(value)
compute sinv = sin(value)
store cos_out[b,t,c] <- cosv
store sin_out[b,t,c] <- sinv

# C Control
parallel idx over B*SEQ*2DIM
guard idx < B*SEQ*2DIM

# H Target Hints
target=triton_ascend
num_warps=4
num_stages=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; host passes precomputed inv_freq and position_angles buffers (already register_buffers, moved to device with the model) plus timestamps, and calls one fused triton kernel returning cos and sin outputs"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the 14-elementwise forward chain (arange, repeat_interleave, broadcast, cat, mul, cos, sin) into a single Triton kernel that reads the precomputed inv_freq and position_angles buffers and timestamps and writes both cos and sin outputs directly","expected_causal_chain":["per-call device kernel count drops from 14 to 1","per-call launch overhead and intermediate tensor allocation disappear","device time per call decreases","wall time per call decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"},{"name":"wall_time_ms","expectation":"decrease"}],"guardrails":["correctness:pass","output structure tuple (cos, sin) unchanged","output shape [4,32,128] fp32 unchanged","inv_freq and position_angles remain register_buffers","ModelNew public contract unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The four recorded failures (winner-tree expert selection, sort-32/sort-64 selection network, dynamic tl.gather compaction, cumsum compaction) are all reduction/selection-network anti-patterns under `groupedtopk` on MLU590-H8; none matches this operator. This operator is a pure elementwise/gather-free chain with no reduction, no selection network, no dynamic gather, and no prefix/compaction; the anti-pattern preconditions do not apply.
- Consulted `references/bottleneck-judgment.md`. device_ratio = 47.78 / (0.581820 * 1000) = 0.082 (< 20%), classifying the operator as host-bound. The intervention targets the launch/routing overhead (14 small kernels per call), which the table classifies as "wrapper routing operations — potentially compressible".
- Guardrail: the fused kernel must NOT recompute `inv_freq` or `position_angles` per forward — those remain `register_buffer`s populated once in `__init__` and passed as kernel inputs, preserving the "model state, not per-forward" semantics confirmed in Phase 0.

## Rationale and Evidence

`report_000.md` establishes the baseline: reference median wall time 0.581820 ms with only 47.78 us of device time spread across 14 kernels per call, a device ratio of ~8.2%. This is the textbook host-bound fragmentation signature: ~92% of wall time is launch, routing, and intermediate-allocation overhead, not device compute.

The forward is a pure elementwise chain (arange → repeat_interleave → broadcast → cat → mul → cos/sin) with no matmul and no reduction. All intermediate tensors (`batch_freqs [4,64]`, `time_freqs [32,64]`, broadcast `[4,32,64]`, concatenated `freqs [4,32,128]`, `angle [4,32]`) are transient. The two non-transient inputs, `inv_freq [32]` and `position_angles [256,64]`, are already `register_buffer`s, so a fused kernel can consume them directly without recomputation.

Because the output is only 2×4×32×128 = 32768 elements, a single 1D Triton kernel indexed over `B*SEQ*2DIM = 16384` programs can decode each element's `(b, t, c)`, select the frequency source by `c < dim` (batch vs time), apply the shared `-timestamps[b,t]*2π` angle, and write both `cos` and `sin` in one pass — collapsing 14 kernels and 6 intermediate allocations into 1 kernel and 2 output allocations. The expected wall improvement is large because the eliminated overhead dominates wall time; the adoption threshold is 5%, and this intervention targets the dominant (host) cost directly.

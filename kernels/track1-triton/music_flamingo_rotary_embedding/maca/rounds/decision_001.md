# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse the full rotary-embedding chain (broadcast/cat/angle-scale/cos/sin) into one direct-launch Triton-MACA elementwise kernel over (B*SEQ, 2*dim) output elements","allowed_changes":["ModelNew.forward","new fused Triton kernel"],"invariants":["ModelNew public contract","output tuple shape dtype device","numerical semantics","non-mutation of timestamps","caller-selected device and current stream","PyTorch fallback for non-benchmark shapes"],"expected_wall_improvement_pct":73.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor timestamps shape=[B,SEQ] dtype=fp32 layout=contiguous memory=global
tensor inv_freq shape=[D2] dtype=fp32 layout=contiguous memory=global
tensor position_angles shape=[MAX_SEQ,2*D2] dtype=fp32 layout=contiguous memory=global
tensor cos_out shape=[B,SEQ,2*D2] dtype=fp32 layout=contiguous memory=global
tensor sin_out shape=[B,SEQ,2*D2] dtype=fp32 layout=contiguous memory=global
scalar D2 dim_half = dim/2
scalar SEQ seq_len = seq_len
scalar MAX_SEQ max_seq_len = max_seq_len
scalar TWOPI two_pi = 2*pi

# O Operations
load ts <- timestamps[b,s]
load ifr <- inv_freq[j]
load tfa <- position_angles[s,j]
compute bf = (b/MAX_SEQ) * ifr
compute half = j < D2
compute f = half ? bf : tfa
compute ang = -ts * TWOPI
compute x = f * ang
compute c = cos(x)
compute s = sin(x)
store cos_out[b,s,j] <- c
store sin_out[b,s,j] <- s

# C Control
parallel (b,s,j) over B*SEQ*2*D2
guard b < B
guard s < SEQ
guard j < 2*D2

# H Target Hints
target=triton_maca
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: forward calls one fused Triton kernel; no cached buffers, no output reuse, no host-side state beyond the unchanged constructor buffers"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the full rotary-embedding chain (broadcast/cat/angle-scale/cos/sin) into one direct-launch Triton-MACA elementwise kernel over (B*SEQ, 2*dim) output elements","expected_causal_chain":["11 host-launched PyTorch kernels collapse to 1 Triton kernel","host launch overhead (~73% of wall) disappears","intermediate tensor materializations are removed","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"candidate_kernel_count_per_call","expectation":"11.0 -> 1.0"},{"name":"candidate_device_us_per_call","expectation":"decrease from 50.95"},{"name":"broadcast_mul_plus_cat_us_per_call","expectation":"~20.9 -> 0.0"},{"name":"fused_triton_kernel_count_per_call","expectation":"== 1.0"},{"name":"cos_sin_allclose","expectation":"pass within atol=rtol=1e-2"}],"guardrails":["correctness:pass","output tuple shape dtype device unchanged","non-mutation of timestamps","caller-selected device and current stream preserved","PyTorch fallback preserved for non-benchmark shapes"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; none of the recorded failures (winner-tree
  argmax, sort selection networks, dynamic `tl.gather`, cumsum compaction) apply — this
  decision introduces no reduction, no on-chip gather, no prefix/compaction, and no
  selection network. The fused kernel is a flat elementwise map (indexing + one
  multiply + one trig pair), which is outside every recorded failure precondition.
- Consulted `triton_maca.md` target profile: the fused kernel uses only `tl.load`,
  `tl.store`, `tl.arange`, and scalar math, all `Supported` on the C500; it uses
  `num_warps=1` (the only `Constrained`-safe value, warp_size=64). No `Unknown`
  primitive (`tl.zeros`, `tl.dot`, block pointers, async copy) is required. The
  kernel avoids `torch.cos`/`torch.sin` and `torch.cat` entirely.
- Direct launch `kernel[(grid,)](...)` is the proven launcher path; `fast_libentry`
  is Unsupported and must not be used. `MACA_PATH=/opt/maca` must be set before
  importing Triton (environment block otherwise, not a kernel failure).

## Rationale and Evidence

`report_000.md` establishes the accepted baseline: wall median 0.190557 ms with
device time 50.95 us/call across 11 kernels/call, giving device_ratio ≈ 0.267. That
means ~73% of wall time is host launch overhead dispatching 11 small PyTorch
elementwise kernels, not device execution. The top device kernels are a broadcast
multiply (`freqs*angle`, ~13.2 us/call), `torch.cat` (~7.7 us/call), and
`cos`/`sin` (~5.3 us/call each); the broadcast-mul plus cat pair alone is ~20.9
us/call of intermediate materialization and launch.

The intervention is a single-kernel fusion: one Triton-MACA elementwise kernel maps
each output element `(b, s, j)` of the two `(4, 32, 128)` outputs to its value,
computing `freqs` from the constructor buffers and `timestamps` in registers and
writing `cos` and `sin` directly. This collapses 11 launches to 1 (removing the
dominant ~73% host overhead) and eliminates every intermediate materialization
(`batch_freqs`, `time_freqs` broadcast, the concatenated `freqs`, and `angle`).
Because the operation is a pure elementwise transform, correctness reduces to exact
indexing math, so the atol/rtol 1e-2 tolerance is trivially met.

Correctness mapping (must reproduce `base.py` exactly):
- `inv_freq[j]` for `j < dim//2` (built once in the constructor); `position_angles`
  already `repeat_interleave(2)`'d in the constructor, so `time_freqs` reads
  `position_angles[s, j]` directly for all `j < 2*dim`.
- `batch_freqs` value = `(b / max_seq_len) * inv_freq[j]`, where the batch index `b`
  divides by `max_seq_len` (NOT `seq_len`) — matching
  `arange(B)/max_seq_len`, then `repeat_interleave(2)` means the same value applies
  for both the even and odd `j` within the batch half.
- Concatenation order along the last dim: the first `dim//2` columns are the batch
  half (`batch_freqs`) and the remaining `dim//2` columns are the position half
  (`position_angles[:seq_len]`), so `f = (j < dim//2) ? bf : tfa`.
- `angle = -timestamps[b,s] * 2*pi`; `freqs *= angle.unsqueeze(-1)` means the scalar
  angle multiplies every `j` column for that `(b, s)`.
- Outputs are `cos(x)` and `sin(x)` of the final scaled `freqs`.

Fallback: `forward` must retain the unchanged pure-PyTorch path from `base.py` and
dispatch to the Triton kernel only when `timestamps` is fp32 on CUDA-compatible
device and the constructor matches the benchmark config (`dim=64`, `max_seq_len=256`).
Any other shape/dtype/device uses the existing PyTorch composition, preserving the
public contract and non-benchmark semantics.

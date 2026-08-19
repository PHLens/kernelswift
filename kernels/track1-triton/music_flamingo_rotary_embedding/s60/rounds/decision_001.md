# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse the forward elementwise/view ops (arange, mul/div, repeat_interleave, broadcast, cat, angle multiply, cos, sin) into a single Triton elementwise kernel that writes both cos and sin output buffers in one launch","allowed_changes":["ModelNew.forward elementwise computation"],"invariants":["ModelNew public contract","output tuple structure and shape [4,32,128] fp32","state_dict keys {inv_freq, position_angles} unchanged","numerical semantics (atol=1e-2, rtol=1e-2, equal_nan=True)"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor timestamps shape=[4,32] dtype=fp32 layout=contiguous memory=global
tensor inv_freq shape=[32] dtype=fp32 layout=contiguous memory=global
tensor position_angles shape=[256,64] dtype=fp32 layout=contiguous memory=global
tensor cos_out shape=[4,32,128] dtype=fp32 layout=contiguous memory=global
tensor sin_out shape=[4,32,128] dtype=fp32 layout=contiguous memory=global
scalar seq_len
scalar max_seq_len

# O Operations
load ts <- timestamps[b,t]
compute angle = -ts * 6.283185307179586
load inv <- inv_freq[k]
compute batch_freq = (b / max_seq_len) * inv
load pa <- position_angles[t,k]
compute freq = select(d < 64, batch_freq, pa)
compute theta = freq * angle
compute c = cos(theta)
compute s = sin(theta)
store cos_out[b,t,d] <- c
store sin_out[b,t,d] <- s

# C Control
parallel index over 16384
guard index < 16384

# H Target Hints
target=triton_gcu
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: no host-side state, allocation, cache, stream, or lifecycle behavior is altered"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the forward elementwise/view ops (arange, mul/div, repeat_interleave, broadcast, cat, angle multiply, cos, sin) into a single Triton elementwise kernel that writes both cos and sin output buffers in one launch","expected_causal_chain":["13 eager topsLaunchKernel launches collapse to 1 fused Triton launch","runtime_launch_us_per_call decreases","intermediate eager tensors disappear","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"runtime_launch_count_per_call","expectation":"decrease"},{"name":"runtime_launch_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output tuple structure and shape [4,32,128] fp32 unchanged","state_dict keys {inv_freq, position_angles} unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; the listed failures (winner tree, sort networks, dynamic gather, cumsum compaction) all stem from reduction/selection-network state bloat on MLU top-k workloads. This workload is pure elementwise (no reduction, no serial loop, no dynamic gather), so none of those preconditions match.
- Key contrast with sparse_pooler s60 fusion failure: that fused kernel carried a serial `range(seq_len)` segment-max reduction whose device penalty exceeded the host-launch savings. This operator has no data-axis reduction and no serial loop — the fusion is a flat elementwise map, so that failure mode does not transfer.
- `tl.cos` and `tl.sin` are NOT listed in the triton_gcu profile's Supported, Constrained, or Unknown tables. They are unproven on this runtime. `tl.exp` (elementwise fp32) IS proven Supported, which shows the MLIR math dialect lowering works, and cos/sin are core members of the same math dialect, so the risk is moderate but real. Coder MUST report a capability-miss (not silently fall back to eager `torch.cos`/`torch.sin`, which would re-split the launch and destroy the fusion) if `tl.cos`/`tl.sin` fail to compile or lower on GCU.
- `num_stages` is Unknown and must not be asserted; `num_warps=1` is the only proven launch configuration.

## Rationale and Evidence

The accepted baseline (`report_000.md`) shows the eager reference issues 13 `topsLaunchKernel` launches per forward call, all from pure elementwise/view ops, with launch overhead ~139.6 us/call ≈ 30% of the 0.465926 ms wall — the largest launch ratio seen among the s60 operators. Collapsing 13 launches to 1 has a direct, attributable mechanism for a >=5% wall win, and additionally removes the intermediate eager tensor reads/writes on device.

The computation is a flat elementwise map with no GEMM and no data-axis reduction. Forward semantics from `base.py`:

1. `batch_freqs[b, d] = (b / max_seq_len) * inv_freq[d // 2]`, shape [4, 64].
2. `time_freqs[t, d] = position_angles[t, d]` = `(t / max_seq_len * 2π) * inv_freq[d // 2]`, sliced to [32, 64] via `position_angles[:seq_len]`.
3. `freqs = cat(batch_freqs[:,None,:], time_freqs[None,:,:], dim=-1)` → [4, 32, 128], columns 0..63 from `batch_freqs` (broadcast over time), columns 64..127 from `time_freqs` (broadcast over batch).
4. `angle = -timestamps * 2π`; `theta = freqs * angle[..., None]`; output `(cos(theta), sin(theta))`.

Fusion scope: `inv_freq` [32] and `position_angles` [256,64] are register buffers computed host-side in `__init__` and are NOT fused — they stay as precomputed inputs loaded by the kernel. Everything from the forward `arange` onward is the fusion target. The kernel maps each output element `(b, t, d)` in the [4,32,128] = 16384-element grid: it loads `timestamps[b,t]`, selects the `batch_freq` vs `time_freq` branch on `d < 64`, multiplies by `angle`, and writes `cos` and `sin` to two output buffers in the single launch. `seq_len` is a Python int passed as a runtime scalar; `max_seq_len` is a module attribute passed as a scalar. With `num_warps=1`, the tile/block size is kept conservative (a 1-D flattened index with a `guard index < 16384`).

Adoption still requires correctness at `atol=1e-2, rtol=1e-2`, the tuple-of-two-tensors `[4,32,128]` fp32 contract, unchanged state_dict keys, and an unrounded median wall improvement ≥5% against `baseline_adapter.py`.

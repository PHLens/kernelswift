# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse the forward elementwise chain (arange/div/repeat_interleave/broadcast/cat/neg/mul/cos/sin) into a single Triton kernel that reads timestamps and the precomputed inv_freq/position_angles buffers and writes both cos and sin outputs, collapsing ~13 device kernel launches per call to 1","allowed_changes":["ModelNew.forward dataflow","fused Triton kernel"],"invariants":["ModelNew public contract","output tuple structure","output dtype and shape","numerical semantics","input not mutated","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":12.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor timestamps shape=[B,S] dtype=fp32 layout=contiguous memory=global
tensor inv_freq shape=[D2] dtype=fp32 layout=contiguous memory=global
tensor position_angles shape=[MAXS,D] dtype=fp32 layout=contiguous memory=global
tensor cos_out shape=[B,S,D2X] dtype=fp32 layout=contiguous memory=global
tensor sin_out shape=[B,S,D2X] dtype=fp32 layout=contiguous memory=global
scalar D2X shape=[1] dtype=int
scalar D shape=[1] dtype=int
scalar MAXS shape=[1] dtype=int
scalar B shape=[1] dtype=int
scalar S shape=[1] dtype=int

# O Operations
compute batch_pos = arange(B) / MAXS           # [B]
compute angle = -timestamps[b,s] * 2pi         # scalar per (b,s)
compute bf64 = (batch_pos[b] * inv_freq[k2]) repeat_interleaved over k2  # [64] = repeat_interleave(batch_pos[b]*inv_freq[32], 2)
compute tf64 = position_angles[s, k]           # [64] (position_angles already interleaved in __init__)
compute freqs = concat(bf64, tf64)             # [128] = cat((batch_freqs, time_freqs), dim=-1)
compute x = freqs * angle                      # angle broadcast over 128 dim
store cos_out[b,s,:] <- cos(x)
store sin_out[b,s,:] <- sin(x)

# C Control
parallel b over B
parallel s over S
parallel k over D
guard b < B
guard s < S
guard k < D

# H Target Hints
target=triton_cuda
num_warps=4
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; inv_freq and position_angles remain precomputed registered buffers, and the fused kernel is launched inside forward with no new host-side state or allocation reuse"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the forward elementwise chain (arange/div/repeat_interleave/broadcast/cat/neg/mul/cos/sin) into a single Triton kernel that reads timestamps and the precomputed inv_freq/position_angles buffers and writes both cos and sin outputs, collapsing ~13 device kernel launches per call to 1","expected_causal_chain":["per-call device kernel count drops from 10.86 to approximately 1-2","host launch and dispatch overhead decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","output tuple structure unchanged","input not mutated"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No entry's preconditions match this operator: the catalog records grouped top-k selection-network, dynamic `tl.gather`, and cumsum compaction failures on an MLU590-H8 runtime, not the BI150 / CoreX fp32 rotary-embedding elementwise chain. The four anti-patterns all concern reduction/selection/gather dataflow that is absent here.
- Target-profile consultation (`prompts/coder_targets/triton_cuda.md`): `tl.arange`, `tl.load`, `tl.store`, `tl.reshape`, `tl.where`, `tl.broadcast_to`, and elementwise math over contiguous fp32 vectors are supported; `num_warps`/`num_stages` are Unknown and must remain non-normative. The fused kernel uses only supported elementwise primitives, and the `num_warps=4` hint is advisory, not a correctness requirement.
- The harness AST loader retains `Import`, `ImportFrom`, `ClassDef`, and `FunctionDef` nodes, so a `@triton.jit`-decorated top-level function is preserved. The `cos`/`sin` computed in-kernel must match `torch.cos`/`torch.sin` to within the harness tolerance (`atol=1e-2, rtol=1e-2`), which is loose for fp32 transcendental functions.
- Fusion is the intended mechanism here (many small elementwise launches inside the change boundary), which is the one case `bottleneck-judgment.md` explicitly endorses for library-kernel fusion.

## Rationale and Evidence

Phase 0 profiler evidence (`rounds/report_000.md`) establishes a host-bound baseline: `baseline_base` scope measured `68.636 us` device time and `10.86` kernels per forward call against `0.353447 ms` wall time, giving `device_ratio = 0.194`. Roughly 80% of wall time is host/launch overhead, and the device work is a long chain of small elementwise kernels (MulFunctor binary `17.69 us`, Cat `10.95 us`, AUnaryFunctor Mul `7.92 us`, sin `7.28 us`, cos `7.13 us`, direct_copy `6.06 us`, BUnaryFunctor Mul `4.03 us`, neg `3.95 us`, arange `3.62 us`). These kernels all correspond to the `forward` elementwise sequence in `base.py`: `arange`, division by `max_seq_len`, `unsqueeze`/broadcast multiplication by `inv_freq`, `repeat_interleave(2)`, `broadcast_tensors`, `cat(dim=-1)`, `neg`/`mul` by `angle`, and `cos`/`sin`. `inv_freq` (length 32) and `position_angles` (256x64) are precomputed in `__init__` and registered as buffers, so they are stable across calls and need no per-call recomputation.

Fusing this chain into a single Triton kernel collapses ~13 launches to 1, directly compressing the dominant host-bound cost. Device work itself is trivial (~68 us), so the expected gain comes from launch/dispatch reduction, not device math. Because host-bound wall time is dominated by per-launch overhead rather than device duration, the improvement is meaningful but conservatively bounded: an expected `12%` wall improvement is reasonable given the launch count drops an order of magnitude while device work is near-constant; the adoption threshold remains the harness `5%` unrounded median. If the remaining wall time after fusion turns out to be harness-fixed (seed/synchronization), subsequent rounds can switch change family per `bottleneck-judgment.md`.

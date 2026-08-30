# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"grid-parallelism"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"partition the 16384-element fused elementwise kernel across a grid of many programs (BLOCK=128, grid=128) instead of a single program with BLOCK=16384, keeping num_warps=1 and one launch that writes both cos and sin","allowed_changes":["ModelNew.forward kernel launch grid"],"invariants":["ModelNew public contract","output tuple structure and shape [4,32,128] fp32","state_dict keys {inv_freq, position_angles} unchanged","numerical semantics (atol=1e-2, rtol=1e-2, equal_nan=True)"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor timestamps shape=[4,32] dtype=fp32 layout=contiguous memory=global
tensor inv_freq shape=[32] dtype=fp32 layout=contiguous memory=global
tensor position_angles shape=[256,64] dtype=fp32 layout=contiguous memory=global
tensor cos_out shape=[4,32,128] dtype=fp32 layout=contiguous memory=global
tensor sin_out shape=[4,32,128] dtype=fp32 layout=contiguous memory=global
tile idx shape=[BLOCK] dtype=int32 memory=register
scalar seq_len
scalar max_seq_len

# O Operations
compute offs = pid * BLOCK + idx
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
parallel pid over 16384/BLOCK
guard offs < 16384

# H Target Hints
target=triton_gcu
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: the launch grid and tile size change inside forward; no host-side state, allocation, cache, stream, or lifecycle behavior is altered"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"partition the 16384-element fused elementwise kernel across a grid of many programs (BLOCK=128, grid=128) instead of a single program with BLOCK=16384, keeping num_warps=1 and one launch that writes both cos and sin","expected_causal_chain":["many programs run concurrently on the device instead of one serial warp","the fused kernel's device execution time collapses toward eager elementwise time","launch stays at 1 and launch overhead stays low","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"runtime_launch_count_per_call","expectation":"remain"},{"name":"runtime_launch_us_per_call","expectation":"remain"}],"guardrails":["correctness:pass","output tuple structure and shape [4,32,128] fp32 unchanged","state_dict keys {inv_freq, position_angles} unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; none of the cataloged failures (winner tree, sort networks, dynamic gather, cumsum compaction) match a pure elementwise map. The Round 1 regression is not a cataloged anti-pattern but a launch-configuration defect: `block = next_power_of_2(16384) = 16384` produced `grid = (1,)`, so a single program with a single warp serially processed all 16384 elements. The device has `multi_processor_count=2`, so this exposed zero parallelism.
- Contrast with sparse_pooler s60: its fused kernel carried a serial `range(seq_len)` segment-max reduction whose device penalty is intrinsic to the reduction and cannot be removed by more programs. This operator is a flat elementwise map — every output element is independent, so partitioning across many programs is a fundamentally valid fix, not a re-run of a failed family.
- `tl.cos` and `tl.sin` are now PROVEN available on this runtime (Round 1 coder probe). No capability-miss, no eager fallback.
- NEW capability risk: `tl.arange` is only proven at extent 16 and extent 4 in the triton_gcu profile. A tile of `BLOCK=128` is an unproven arange extent. Coder MUST report a capability-miss if `tl.arange(0, 128)` fails to compile/lower, and may then reduce BLOCK toward a proven extent (e.g. BLOCK=16, grid=1024) as a non-normative fallback.
- `tl.program_id` is only proven on axis 0. This decision uses a 1-D grid with in-kernel index decomposition (b, t, d recovered from the flattened offset), NOT a 3-D grid. `num_warps=1` remains the only proven launch configuration; `num_stages` is Unknown and must not be asserted.

## Rationale and Evidence

Round 1 (`report_001.md`) falsified the fusion hypothesis for one specific reason, not for the fusion idea itself: correctness passed and the launch-collapse mechanism was confirmed (`runtime_launch_count_per_call` 13 -> 1, `runtime_launch_us_per_call` 139.38 -> 9.70 us), but wall time regressed to 5.162427 ms vs 0.464657 ms baseline (-1010.99%). The evidence names the cause precisely: `block = triton.next_power_of_2(16384) = 16384` and `grid = (1,)`, so the fused kernel ran as a single program with `num_warps=1` serially over all 16384 elements. The ~5.15 ms device execution is the serial single-warp body cost, not an intrinsic property of elementwise fusion — the eager library ops are internally highly parallel across the same 16384 elements.

The intervention is a launch-grid repair, not a new algorithm: keep the identical fused elementwise kernel body, but set `BLOCK = 128` and `grid = (128,)`, so 128 programs each process 128 elements concurrently. `num_warps=1` is retained. Each program keeps the same flattened-index decomposition (`b = offs // (seq_len*2D)`, `t = rem // (2D)`, `d = rem % (2D)`) with a `guard offs < 16384`.

Expected outcome: the fused kernel's device time collapses toward the eager elementwise time, while the confirmed ~129.7 us launch saving is preserved (still 1 launch). This is expected to clear the >=5% wall threshold against `baseline_adapter.py` (0.464657 ms).

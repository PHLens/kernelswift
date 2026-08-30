# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_mhc_head_compute_mix_backward_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the adoption threshold: the operator is already fused to a single Triton kernel (device 14.692 us/call, device_ratio 0.074) and the remaining ~93% of wall time is harness-fixed per-iteration set_seed + torch.cuda.synchronize overhead outside the candidate's allowed change boundary","allowed_changes":[],"invariants":["ModelNew public contract","forward signature (input_mix,mhc_scale,mhc_base,grad_out)->(grad_input_mix,grad_mhc_scale,grad_mhc_base)","output tuple structure, shapes [2,1024,4]/[1]/[4], all fp32","sigmoid-backward numerical semantics including the two reduction contracts","input tensors not mutated","caller-selected device and current stream preserved","base.py and the harness (set_seed/sync_devices timing floor) are immutable"],"expected_wall_improvement_pct":0.0}
```

## Unified Sketch

N/A: aborted

## Host Plan

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Evaluation Contract

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; no entry preconditions match this operator. The catalog records grouped-topk selection-network failures (`tl.gather` compaction, full-sort selection networks, cumsum compaction) on an MLU590-H8 runtime — reduction/selection dataflow paths absent from this single fused sigmoid-backward kernel over `[2,1024,4]` fp32. No catalog entry names a remaining compressible path.
- The two remaining host-side candidates were assessed against `references/bottleneck-judgment.md` and `references/invariants.md`:
  - `torch.zeros` accumulator initialization (`FillFunctor`, ~7.276 us/call, ~half of device time): this is device-side and would be eliminated by a persistent zero-initialized buffer or in-kernel zeroing. But it is only `7.276 / 198.597 ≈ 3.7%` of wall time, below the 5% adoption threshold even if fully removed. Persistent buffer reuse additionally violates the harness's per-call fresh-tensor comparison semantics and the "input not mutated" invariant, and introduces Host Plan lifecycle/compatibility risk for no attributable ≥5% gain.
  - Output-allocation reuse / launcher reduction: the remaining host time is the harness's per-iteration `set_seed` (`torch.manual_seed` + `manual_seed_all`) and `sync_devices` (`torch.cuda.synchronize()`), both classified **Fixed for the regime** (user-owned harness, immutable) by `bottleneck-judgment.md`. The candidate-boundary host work (one `torch.empty` + two `torch.zeros` allocations + one Triton launch) is a few microseconds, far below the ~9.93 us (5% of 198.597 us) threshold.

## Rationale and Evidence

Round 001 kernel fusion was accepted with a `43.11%` wall improvement (`0.349112 ms → 0.198597 ms` unrounded median), collapsing per-call kernel count from `9.74` to `~2.96` and device time from `186.057 us/call` to `14.692 us/call`. The accepted candidate's profiler evidence (`rounds/report_001.md`) shows:

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000)
             = 14.692 / 198.597 ≈ 0.074
```

Only ~7.4% of wall time is device kernel time; ~92.6% is host/launch/synchronization overhead. The operator is now **host-bound**, and the device side is already at its theoretical minimum: the entire sigmoid-backward chain (elementwise `z -> sigmoid -> grad_z -> grad_input_mix` plus the two on-chip reductions for `grad_mhc_base [4]` and `grad_mhc_scale [1]`) executes in a single fused Triton kernel (`_mhc_head_compute_mix_backward_kernel`, `7.416 us/call`) with no intermediate materialization and no standalone reduction kernels. The input is minuscule (`[2,1024,4]` = 8192 elements), so there is no redundant device dataflow, math, or launch count left to compress.

The Verifier's `evidence_for_next_round` states the only remaining device item is the `torch.zeros` accumulator initialization (`FillFunctor`, ~`7.276 us/call`, ~half of the remaining `14.692 us/call` device time), which a future round could target — but at `3.7%` of wall it cannot clear the 5% threshold, and persistent-buffer reuse violates the harness's fresh-tensor semantics and the "input not mutated" invariant.

The remaining host time is **harness-fixed**, not candidate-compressible. The harness `time_forward` loop runs `set_seed(seed)` → `forward` → `sync_devices()` per timed sample. Both `set_seed` (which calls `torch.manual_seed` plus `manual_seed_all` on every accelerator) and `sync_devices` (which calls `torch.cuda.synchronize()`) are user-owned harness costs classified as **Fixed for the regime** by `bottleneck-judgment.md`, and are immutable under `references/invariants.md` ("base.py is user-owned and immutable", "the harness ... are part of measurement_fingerprint", "Do not optimize base.py or alter the harness to manufacture a speedup"). The candidate-boundary host cost that remains (one `torch.empty` plus two `torch.zeros` allocations plus one Triton launch) is a few microseconds against a ~198.6 us wall dominated by that fixed seed+synchronize floor — well below the ~9.93 us that 5% would require.

This is a **measurement-bound stop**: the accepted report and completed-round evidence show device work below the stated bound (device_ratio 0.074) and the remaining host time is harness-fixed. No falsifiable intervention with an expected ≥5% unrounded median wall improvement can be justified. This matches the documented precedent (task 5 rotary Round 002, and the same-family forward `mhc_head_compute_mix` Round 002, both aborted after a successful Round 001 kernel fusion left the operator host-bound). The canonical implementation remains `triton_mhc_head_compute_mix_backward_001.py`.

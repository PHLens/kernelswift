# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_mhc_head_compute_mix_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention clears the adoption threshold: the operator is already fused to a single kernel (device 12.996 us/call, device_ratio 0.0755) and the remaining ~92.5% of wall time is harness-fixed per-iteration seed + device-synchronization overhead outside the candidate's allowed change boundary","allowed_changes":[],"invariants":["ModelNew public contract (hc_mult=4, sinkhorn_iters=20, eps=1e-6)","forward signature (mixes,hc_scale,hc_base)->(pre,post,comb)","output tuple structure, shapes pre/post [2,8,4] and comb [2,8,4,4], all fp32","exact Sinkhorn numerical semantics including asymmetric eps placement","input tensors not mutated","caller-selected device and current stream preserved","base.py and the harness (set_seed/sync_devices timing floor) are immutable"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. No entry preconditions match this operator: the catalog records grouped-topk selection-network failures (`tl.gather` compaction, full-sort selection networks, cumsum compaction) on an MLU590-H8 runtime. Those are reduction/selection dataflow paths absent from this fixed 20-round row/column alternating normalization over a tiny `[16,4,4]` fp32 matrix. No catalog entry names a remaining compressible path here.
- The remaining host-side options (output-allocation reuse, device-context removal) were assessed against `references/bottleneck-judgment.md` and `references/invariants.md`: the harness's `set_seed` (manual_seed + manual_seed_all) and `sync_devices` (`torch.cuda.synchronize()`) run once per timed iteration and are classified **Fixed for the regime** (user-owned harness, immutable). Caching the three `torch.empty` output buffers in `forward` would only remove a few microseconds of allocation against a ~184 us wall dominated by that fixed floor, far below the 5% adoption threshold, and introduces Host Plan lifecycle/compatibility risk for no attributable gain. No falsifiable ≥5% intervention remains.

## Rationale and Evidence

Round 001 was accepted with an `87.17%` wall improvement (`1.433128 ms → 0.183889 ms` unrounded median), collapsing per-call kernel count from `132.88` to `1.0` and device time from `924.79 us/call` to `12.996 us/call`. The accepted candidate's profiler evidence (`rounds/report_001.md`) shows:

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000)
             = 12.996 / 183.889 ≈ 0.0755
```

Only ~7.5% of wall time is device kernel time; ~92.5% is host/launch/synchronization overhead. The operator is now **host-bound**, and — critically — the device side is already at its theoretical minimum: the entire computation (two sigmoids, one stable-softmax `exp`, and 20 rounds of row/column alternating normalization over the `[16,4,4]` `comb` matrix) executes in a single fused Triton kernel with a 16-program grid, keeping the `[4,4]` tile entirely in registers with no global-memory round-trip between rounds. The input data is minuscule (`mixes[2,8,24]`, `comb[16,4,4]` = 256 elements per matrix), so there is no redundant device dataflow, math, or launch count left to compress.

The Verifier's `evidence_for_next_round` states this explicitly: "Candidate device_ratio dropped to `0.0755` ... Further device-side fusion has limited headroom; the remaining bottleneck is the host-side per-call overhead and the single-kernel launch."

The remaining host time is **harness-fixed**, not candidate-compressible. The harness `time_forward` loop runs `set_seed(seed)` → `forward` → `sync_devices()` per timed sample. Both `set_seed` (which calls `torch.manual_seed` plus `manual_seed_all` on every accelerator) and `sync_devices` (which calls `torch.cuda.synchronize()`) are user-owned harness costs classified as **Fixed for the regime** by `bottleneck-judgment.md`, and are immutable under `references/invariants.md` ("base.py is user-owned and immutable", "the harness ... are part of measurement_fingerprint", "Do not optimize base.py or alter the harness to manufacture a speedup"). The candidate-boundary host cost that remains (three `torch.empty` output allocations plus input `.to(float32)` casts) is a few microseconds against a ~184 us wall dominated by that fixed seed+synchronize floor — an output-buffer reuse intervention would yield well under 5% and cannot clear the adoption threshold.

This is a **measurement-bound stop**: the accepted report and completed-round evidence show device work below the stated bound and the remaining host time is harness-fixed. No falsifiable intervention with an expected ≥5% unrounded median wall improvement can be justified. This matches the documented precedent (task 5 / rotary Round 002: a successful Round 001 kernel fusion left the operator host-bound and Round 002 aborted). The canonical implementation remains `triton_mhc_head_compute_mix_001.py`.

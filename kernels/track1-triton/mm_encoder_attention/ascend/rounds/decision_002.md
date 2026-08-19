# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention clears the 5% adoption threshold; wall time is host-bound (device_ratio ~0.30-0.33) with harness-fixed launch/synchronization dominating, the native flash-attention device path is already near-optimal, and the Triton rewrite only moves device time that does not move wall","allowed_changes":[],"invariants":["base.py is immutable","ModelNew(num_heads=8, head_size=64, num_kv_heads=8) public contract","output Tensor[2,83,512] fp16","numerical semantics within atol=1e-2, rtol=1e-2","harness measurement fingerprint unchanged"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. The catalog entries (winner tree, sort networks, dynamic gather, cumsum compaction) are grouped-topk/MLU-specific and do not directly match, but the meta-pattern they record — "a source-level simplification lowers to duplicated or more-expensive device state on the matched compiler" — is directly relevant: Round 001's Triton materialized attention lowered to a single 104 us kernel, 4.5x the native FA kernel's 23.35 us, the same "Triton compute is more expensive than the native path on this backend" failure mode.
- The verifier's `tl.dot` suggestion is rejected against the campaign's own negative precedent: earlier flexattention `tl.dot` attempts on Ascend showed a host-side penalty (a ~55 us launch penalty is the cited lesson), and the profile proves only `(16,16)@(16,16)` fp32 `tl.dot`. A `tl.dot` rewrite would optimize device time only, which the host-bound wall will not reflect, while risking the known host penalty.

## Rationale and Evidence

Round 001 report establishes two decisive facts. First, the layout-fusion mechanism was structurally correct — kernel count collapsed 6.78 → 1.0 and all four transpose/inplace-copy kernels were eliminated — yet device_us_per_call fell only 14.8 us (118.94 → 104.15), because the single Triton kernel costs 104.15 us on device versus the native flash-attention kernel's 23.35 us (4.5x). Second, and more fundamentally, wall time is host-bound: device_ratio is ~0.30-0.33, so roughly 70% of the ~0.35 ms wall is host-side launch, `torch.empty` allocation, `contiguous()` no-op checks, and the harness's per-sample `sync_devices()` — all of which are harness-fixed (the measurement fingerprint) or already minimal after the kernel-count collapse to 1.

This means no kernel-level intervention can reach the 5% wall threshold. Even eliminating device time entirely would bound the wall improvement at the ~30% device fraction, and the only remaining device-side idea (`tl.dot`) targets device compute that wall does not reflect, while carrying a known negative host penalty from this campaign's prior flexattention attempts. On the host side, the only compressible items (launch count) were already removed in Round 001; the residual host time is the harness-fixed seed setup and `sync_devices()` that `bottleneck-judgment.md` classifies as "fixed for the regime".

This matches the pattern the prior five operators in this campaign converged on: wall time is dominated by fixed Triton launch/harness overhead, and kernel code cannot eliminate it. The native flash-attention path (23.35 us FA + ~62 us layout + ~32 us sync = ~119 us device) is already near-optimal on device, and the Triton alternative is strictly more expensive there (104 us) while providing no wall benefit. No falsifiable intervention with an expected ≥5% unrounded wall improvement remains. Stop reason: host-bound wall with harness-fixed overhead; native backend already near-optimal; Triton rewrite proven unable to move wall.

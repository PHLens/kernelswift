# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"candidate_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the 5% adoption threshold: device_ratio is 0.0224 and the residual ~392us wall is dominated by harness-fixed set_seed/synchronize plus a single fixed Triton launch; the only compressible host work (three tiny torch.empty allocations and no-op .to(fp32).contiguous()) amounts to a few microseconds, far below the ~19.6us needed for 5%","allowed_changes":[],"invariants":["ModelNew public contract and forward signature","output tuple structure, fp32 dtype, and shapes","exact numerical semantics","harness and measurement fingerprint unchanged"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`; no entry invalidates this stop. The catalog records device-side lowering failures (winner trees, full sort networks, dynamic gather, cumsum compaction) on a grouped top-k MLU workload, unrelated to a small 4x4 Sinkhorn kernel now reduced to a single 8.8us launch.
- Consulted `references/bottleneck-judgment.md`: with `device_ratio = 0.0224` (< 5%) and wall time "stuck" after the dominant fusion win, the operator is a `measurement-bound candidate`. The required stop condition is that normalized device work is below the bound and the residual host time is harness-fixed — which report_001 establishes (device 8.8us vs wall 392us; residual host is the harness's per-sample `set_seed` + `synchronize` and a single fixed Triton launch).
- The `triton_ascend` target profile lists `fast_libentry` as Unknown (no proven fast-launcher path), so there is no evidence-backed mechanism to reduce the single-kernel launch floor. No `Supported` primitive can lower it.

## Rationale and Evidence

Report 001 is authoritative and shows the sinkhorn-loop-fusion intervention fully realized its mechanism: `kernel_count_per_call` 136 → 1, `device_us_per_call` 282.4 → 8.8 us, wall 3.527 → 0.392 ms (+88.88%). The remaining wall is now dominated by host time that is not reachable by any falsifiable ≥5% intervention.

Quantitative reasoning against the 5% threshold (5% of 392 us ≈ 19.6 us):

1. **Kernel-only**: device time is 8.8 us out of 392 us (device_ratio 0.0224). Even halving device time saves ~4.4 us, far below 19.6 us. There is no remaining device-side bottleneck; the Sinkhorn loop is already a single `static_range` loop with `tl.sum`/`tl.max` reductions.

2. **Host Plan (allocation reuse / output caching)**: the forward wrapper performs three `torch.empty` allocations (16x4, 16x4, 16x4x4 = 1.5 KB total) plus `.to(torch.float32)` and `.contiguous()` on inputs. The inputs (`mixes`, `hc_scale`, `hc_base`) are already fp32 and contiguous, so `.to(fp32)` and `.contiguous()` return `self` (no copy — zero cost). The three allocations are serviced by the caching allocator in ~3–6 us combined. Caching them removes only that ~3–6 us, not the kernel launch, not the harness synchronization. This cannot reach 19.6 us.

3. **Launch floor**: the single Triton kernel launch dispatch is a fixed cost that buffer caching does not eliminate, and the `triton_ascend` profile records `fast_libentry` as Unknown (no proven fast-launcher path to reduce it).

4. **Harness-fixed**: the benchmark timing loop performs `set_seed(seed)` and `sync_devices()` per sample; these are part of the measurement fingerprint and are immutable by invariant ("Do not optimize base.py or alter the harness to manufacture a speedup"). They are the dominant residual host cost and are not compressible from the candidate side.

The convergence pattern across this campaign's prior operators confirms this: once fusion drops `device_ratio` to the low single digits, the residual wall is the fixed Triton launch plus harness synchronization, which kernel and host changes cannot eliminate. Therefore no falsifiable intervention is expected to clear the 5% adoption threshold, and the round is concluded as an abort with stop reason `measurement-bound` (residual host time is harness-fixed).

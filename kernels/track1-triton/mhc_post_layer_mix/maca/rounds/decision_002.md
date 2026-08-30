# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_mhc_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"memory-bound","intervention":"no stable intervention clears the 5% adoption threshold: the single fused kernel is already at the HBM bandwidth floor and the residual host time is harness-fixed or sub-threshold","allowed_changes":[],"invariants":["ModelNew public contract","output shape dtype device","fp32 accumulate then bf16 cast","input non-mutation","caller-selected device and current stream"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md` and `references/bottleneck-judgment.md`. The recorded anti-pattern entries (winner tree, sort networks, dynamic `tl.gather` compaction, cumsum compaction) concern MLU grouped top-k selection and are inapplicable to this memory-bound streaming kernel; none names a failure that this round would repeat.
- The `num_warps` and `num_stages` primitives are marked Constrained/Unknown on the `triton_maca` profile: only `num_warps=1` is proven. Any proposal to raise `num_warps` to 2/4 is a capability-miss risk with no supporting probe, and on a memory-bound kernel additional warps do not reduce bytes moved.
- Every remaining device-side knob (BLOCK size, index-decode strength reduction, warp count) is a compute/occupancy tweak on a kernel whose cost is dominated by irreducible ~170 MB of global-memory traffic. `bottleneck-judgment.md` requires a named, falsifiable, ≥5% mechanism; no such mechanism remains.

## Rationale and Evidence

Round 001 (`rounds/report_001.md`) reduced wall from 7.633507 ms to 0.241083 ms (96.84%) by collapsing six kernels into one and eliminating the badly-sized tf32 GEMM. The accepted kernel is a single `_mhc_post_layer_mix_fused_kernel` at 168.56 us/call, with `device_ratio = 0.699`.

The remaining device time is already at the memory-bandwidth floor. The op writes a 41,943,040-element bf16 output (~80 MB) and reads the four bf16 residual planes (~80 MB) plus `x` (~10 MB) and small fp32 coefficient tensors, for roughly ~170 MB of unavoidable global-memory traffic. At 168.56 us that is ~1 TB/s effective bandwidth — already close to a C500-class HBM ceiling for a streaming kernel. There is no redundant computation, no wasted K-tile, and no intermediate round-trip left to remove; the write of 80 MB and the read of ~90 MB are irreducible by the operator contract.

The candidate-specific knobs do not clear the 5% threshold of 12.05 us (0.241083 ms × 5%):

1. **BLOCK 1024→2048/4096** only reduces grid from 40960 to 20480/10240 programs. Launch overhead is already amortized across 168 us of memory streaming; the change is a marginal occupancy tweak, not an algorithmic or bandwidth gain, and cannot plausibly save >12 us.
2. **num_warps 1→2/4** is Unknown on this profile (capability-miss risk with no probe), and for a memory-bound kernel more warps do not reduce traffic.
3. **Index-decode div/mod** is compute overhead on a memory-bound kernel; strength reduction is cosmetic.
4. **Output allocation (one ~80 MB `torch.empty`)** is candidate-owned host time, but caching it would be a host change requiring a full Host Plan with per-instance ownership, cache-key, invalidation, and concurrency semantics, and it saves at most a fraction of the ~72 us host slice — below the 12.05 us threshold once harness-fixed `set_seed` + `sync_devices` are excluded.

The remaining ~30% of wall (~72 us) decomposes into a single kernel launch, one output allocation, and harness-fixed `set_seed`/`sync_devices` cost. `bottleneck-judgment.md` classifies seed setup and harness synchronization as fixed for the regime; the compressible host remainder (one launch + one alloc) is well under the adoption threshold and would not move unrounded median wall by 5%.

No Verifier-backed observation names a ≥5% mechanism. The primary bottleneck (tf32 GEMM) is fully eliminated, and the operator is now memory-bound at the bandwidth floor. Per the workflow contract, this justifies an honest stop rather than a speculative micro-tuning hypothesis.

# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"003","reference_implementation":"candidate_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"no stable kernel-only intervention clears the 5% wall-time adoption threshold: device time no longer controls wall time (device_ratio ~0.70), the remaining ~0.26 ms wall-device gap is harness-fixed per-sample synchronization and seed setup, and every proven device-side lever is exhausted","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","fp32 accumulation before single bf16 cast","benchmark and measurement-fingerprint semantics"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`, `references/bottleneck-judgment.md`, and
  `prompts/coder_targets/triton_ascend.md`. No remaining falsifiable five-percent
  kernel-only hypothesis survives the accumulated evidence.
- `kernel-fusion` (R001) already captured the dominant win (+264%, 6 kernels → 1).
  `kernel-tuning` (R002) confirmed the kernel is latency-bound and the BLOCK_C/
  num_warps knob is directionally correct (device −3.9%), but wall time did not
  move (−0.58%, noise), which falsifies the "device↓ → wall↓" causal link.
- `tl.dot` remains ruled out: the contraction dim `m=4` is below the probed
  `(16,16)@(16,16)` shape, and padding would multiply MACs 8–16x on a
  memory-bound kernel. A 2D-grid flattening was already measured at 0.926x in
  coder_result_001 attempt #2.
- The residual ~0.26 ms host/sync gap is harness-fixed: `bottleneck-judgment.md`
  classifies "seed setup in user-owned harness" and "harness device
  synchronization" as "Fixed for the regime", and they are part of the
  measurement fingerprint. The only unproven candidate-side lever (`fast_libentry`
  launcher) is `Unknown` on triton_ascend and outside the kernel-only family.

## Rationale and Evidence

The operator is at its practical floor under this harness. report_000 established
device-bound baseline (device_ratio ~0.96, 6 kernels). report_001 (accepted)
collapsed those six kernels into one fused kernel for +264% wall improvement
(3.198 → 0.880 ms). report_002 then showed the remaining wall time is no longer
device-controlled: a genuine −3.9% device-time gain (620.84 → 596.92 us) produced
−0.58% wall change (within noise), because device_ratio is ~0.68–0.71 and ~30%
of wall is a harness-fixed host/sync gap (~0.26 ms of per-sample `sync_devices()`
plus `set_seed` and one launch). Every proven kernel-side lever — fusion, block/
warp tuning, matmul lowering, layout — is now exhausted or rejected. The only
remaining compressible target (the unproven `fast_libentry` launcher) is outside
the kernel-only change family and unverified on this runtime, so no falsifiable
intervention with expected wall improvement ≥5% remains. This matches the
campaign-wide pattern in which groupedtopk, flexattention, fused_moe,
sparse_pooler, music_rotary, and mm_encoder_attention all aborted on a fixed
Triton launch/host floor. Stop reason: `host-bound-floor` — wall time is
harness-fixed-host dominated and further device tuning is sub-threshold.

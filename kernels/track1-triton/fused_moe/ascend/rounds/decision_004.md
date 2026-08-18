# Decision 004

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"004","reference_implementation":"triton_fused_moe_003.py","reference_report":"rounds/report_003.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention clears the 5% adoption threshold; the workload is firmly host-bound with device near structural floor, allocation reuse exhausted, and the remaining host time is fixed backend launch/dispatch overhead","allowed_changes":[],"invariants":["ModelNew public constructor and forward contract","output shape [83,128] fp16","softmax+topk+renormalize routing and weighted top-k reduce semantics","benchmark measurement semantics"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. The recorded failures concern MLU590 device-side top-k selection; they are not relevant to this terminal host-bound state.
- Consulted `references/bottleneck-judgment.md`. The candidate device_ratio is ~0.07 (< 20%), classifying the workload as firmly host-bound; the only remaining compressible host lever (per-call output allocation) was already removed in Round 3, and the remaining ~340 us of host time is fixed Triton launch/dispatch overhead plus harness synchronization.
- Consulted the sibling Ascend campaign terminations: groupedtopk-ascend decision_003 and flexattention-ascend decision_004 both aborted at this exact terminal state after exhausting kernel fusion + allocation reuse and measuring a ~107 us fixed Triton launch overhead with `fast_libentry`/stream/context all Unknown on Ascend. flexattention-ascend decision_004 additionally rejected a device-side `tl.dot` change that regressed wall -8.34% via a Cube-unit host penalty. The same evidence bounds this campaign.
- Consulted `prompts/coder_targets/triton_ascend.md`. The remaining named host lever (moving the two `w1/w2 .to(dtype)` fp16 casts out of `forward`) is only ~4.8 us device plus 2 host launches (~1.5-3% of wall), below the 5% threshold, and is correctness-delicate because the harness calls `load_state_dict` after construction. No other Supported primitive or host-side path offers a falsifiable >=5% win.

## Rationale and Evidence

Three proceeding rounds were accepted: Round 1 kernel fusion (126 -> 12 kernels, wall 7.159 -> 0.5696 ms, +92.7%), Round 2 routing-in-kernel fusion (12 -> 3 kernels, 0.5696 -> 0.3690 ms, +35.9%), and Round 3 output-buffer allocation reuse (0.4003 -> 0.3735 ms, +6.70%). The cumulative result is ~19-21x over the baseline (7.159 ms -> 0.3735 ms wall; the harness accuracy pass reports 21.3x vs base.py).

After Round 3, the candidate is at its structural floor: 26.622 us device per call (one fused per-token Triton kernel ~21.9 us plus two `w1/w2.to(dtype)` fp16 casts ~4.8 us) against 0.373490 ms wall, device_ratio ~0.07. The workload is firmly host-bound. The remaining ~340 us of host time is dominated by fixed Triton launch/dispatch overhead and harness synchronization, not by any compressible, correctly-attributable per-forward work.

The one remaining named lever — moving the two fp16 casts out of `forward` — is ~1.5-3% of wall (4.8 us device + 2 host launches), which does not clear the 5% adoption threshold and is correctness-delicate due to the harness's post-construction `load_state_dict`. No device-side change clears 5% either: device is only 7% of wall and already minimal, and the proven device-side alternative (`tl.dot` Cube path) regressed wall on this identical runtime (flexattention-ascend Round 3, -8.34%) by adding host launch penalty. Reducing the launch grid would trade device parallelism for no host-launch reduction, and `fast_libentry`/stream/context remain Unknown on Ascend (groupedtopk-ascend Round 3 measured ~107 us fixed launch overhead before aborting).

This is the same terminal state that terminated both sibling Ascend campaigns. No stable, falsifiable intervention with expected >=5% wall improvement remains; the correct decision is to halt. Final cumulative result: fused_moe-ascend reaches ~19-21x wall-speedup over the baseline across three accepted rounds (kernel fusion, routing fusion, allocation reuse).

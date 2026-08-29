---
schema_version: 1
id: pattern-device-win-wall-loss
title: Device win with wall-time loss
type: pattern
audiences: [designer]
authority: advisory
summary: Preserve the distinction between a kernel-time result and end-to-end wall behavior.
targets: [ascend]
target_match: unknown
languages: [cpp, python, triton]
kernel_types: []
techniques: [tiling]
hardware_features: []
tags: [device-win-wall-loss, host-bound, synchronization]
symptoms: [device-win-wall-loss]
sources: [source-local-ascend-flexattention-round-003, source-local-s60-flexattention-round-001, source-local-s60-mm-encoder-attention-round-002, source-local-s60-sparse-pooler-round-001, source-mskl-user-guide-f9fbf4d2]
related: []
prerequisites: []
version_sensitive: []
candidate_techniques: [technique-tiling-and-work-partitioning]
observations:
  - id: observation-kernel-time-is-not-wall-time
    text: The reviewed MSKL example reports kernel execution time for tuning configurations but does not report a synchronized end-to-end application wall-time field.
    source_id: source-mskl-user-guide-f9fbf4d2
    locator: artifact user guide lines 308-355
    evidence_level: inferred
    reproduction: concept
    targets: [ascend]
    target_match: unknown
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [absence of wall-time evidence does not establish either a wall win or a wall loss]
examples:
  - id: example-flexattention-device-win-wall-loss-round-003
    role: counterexample
    subtype: device-wall-mismatch
    source_id: source-local-ascend-flexattention-round-003
    locator: reviewed proposal observations and transfer boundaries
    evidence_level: source-reported
    reproduction: runnable
    target_id: ascend910b4
    implementation_profile_id: triton_ascend
    profile_authority: historical-noncanonical
    runtime_fingerprint: triton-3.2.0 torch-npu-2.7.1.post4
    operator_family: flexattention
    shape:
      HEADS: 8
      HEAD_SIZE: 64
      KV_HEADS: 8
      TOKENS: 83
    dtype: fp16
    terminal_classification: no-improvement
    comparability: historical-local
    measurement_fingerprint: c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8
    baseline_id: null
    candidate_id: null
    observed:
      - {metric: correctness_pass, value: true, statistic: exact, unit: boolean}
      - {metric: device_improvement_pct, value: 55.80633070472635, statistic: exact, unit: percent}
      - {metric: device_time_ms, value: 0.0240532, statistic: exact, unit: milliseconds}
      - {metric: wall_improvement_pct, value: -8.344714789147998, statistic: exact, unit: percent}
      - {metric: wall_time_ms, value: 0.32128, statistic: median, unit: milliseconds}
    transfer_boundary: exact Ascend910B4, triton_ascend, Triton 3.2.0 / torch_npu 2.7.1.post4, tokens=83, heads=8, head_size=64, kv_heads=8, fp16 input/output with fp32 accumulation, original synchronization policy, host path, Round 003 measurement, and harness only; reference device time 0.0544268 ms and wall median 0.296535 ms
    reconsider_when: [binding:missing-vnext-artifact, sketch:missing-vnext-artifact, verdict:missing-vnext-artifact]
  - id: example-flexattention-device-win-wall-loss-s60-round-001
    role: counterexample
    subtype: device-wall-mismatch
    source_id: source-local-s60-flexattention-round-001
    locator: reviewed proposal observations and transfer boundaries
    evidence_level: source-reported
    reproduction: runnable
    target_id: s60
    implementation_profile_id: triton_gcu
    profile_authority: historical-noncanonical
    runtime_fingerprint: triton-3.6.0 triton_gcu-3.6.0+1.0.20260722 torch-2.10.0+cpu torch_gcu-2.10.0+3.8.0.2
    operator_family: flexattention
    shape:
      HEAD: 8
      HEAD_SIZE: 64
      TOKENS: 83
    dtype: fp16
    terminal_classification: no-improvement
    comparability: historical-local
    measurement_fingerprint: null
    baseline_id: null
    candidate_id: null
    observed:
      - {metric: correctness_pass, value: true, statistic: exact, unit: boolean}
      - {metric: wall_improvement_pct, value: -6.3954, statistic: exact, unit: percent}
      - {metric: wall_time_ms, value: 0.266835, statistic: median, unit: milliseconds}
    transfer_boundary: exact S60 (Enflame GCU), triton_gcu, triton 3.6.0, tokens=83, heads=8, head_size=64, fp16, causal attention, Round 001 measurement, and harness only
    reconsider_when:
      - target, profile, runtime, shape, or dtype scope changes
  - id: example-mm-encoder-attention-device-win-wall-loss-s60-round-002
    role: counterexample
    subtype: device-wall-mismatch
    source_id: source-local-s60-mm-encoder-attention-round-002
    locator: reviewed proposal observations and transfer boundaries
    evidence_level: source-reported
    reproduction: runnable
    target_id: s60
    implementation_profile_id: triton_gcu
    profile_authority: historical-noncanonical
    runtime_fingerprint: triton-3.6.0 triton_gcu-3.6.0+1.0.20260722 torch-2.10.0+cpu torch_gcu-2.10.0+3.8.0.2
    operator_family: mm-encoder-attention
    shape:
      BATCH: 2
      HEAD: 8
      HEAD_SIZE: 64
      SEQ: 83
    dtype: fp16
    terminal_classification: no-improvement
    comparability: historical-local
    measurement_fingerprint: null
    baseline_id: null
    candidate_id: null
    observed:
      - {metric: correctness_pass, value: true, statistic: exact, unit: boolean}
      - {metric: wall_improvement_pct, value: -9.3151, statistic: exact, unit: percent}
      - {metric: wall_time_ms, value: 0.275038, statistic: median, unit: milliseconds}
    transfer_boundary: exact S60 (Enflame GCU), triton_gcu, triton 3.6.0, batch=2, heads=8, head_size=64, seq=83, fp16, bidirectional MHA, Round 002 measurement, and harness only
    reconsider_when:
      - target, profile, runtime, shape, or dtype scope changes
  - id: example-sparse-pooler-device-win-wall-loss-s60-round-001
    role: counterexample
    subtype: performance
    source_id: source-local-s60-sparse-pooler-round-001
    locator: reviewed proposal observations and transfer boundaries
    evidence_level: source-reported
    reproduction: runnable
    target_id: s60
    implementation_profile_id: triton_gcu
    profile_authority: historical-noncanonical
    runtime_fingerprint: triton-3.6.0 triton_gcu-3.6.0+1.0.20260722 torch-2.10.0+cpu torch_gcu-2.10.0+3.8.0.2
    operator_family: sparse-pooler
    shape:
      HIDDEN: 768
      TOKENS: 83
      VOCAB: 30522
    dtype: fp32
    terminal_classification: no-improvement
    comparability: historical-local
    measurement_fingerprint: null
    baseline_id: null
    candidate_id: null
    observed:
      - {metric: correctness_pass, value: true, statistic: exact, unit: boolean}
      - {metric: wall_improvement_pct, value: -302.19, statistic: exact, unit: percent}
      - {metric: wall_time_ms, value: 3.447952, statistic: median, unit: milliseconds}
    transfer_boundary: exact S60 (Enflame GCU), triton_gcu, triton 3.6.0, tokens=83, hidden=768, vocab=30522, fp32, segment-max reduction, Round 001 measurement, and harness only
    reconsider_when:
      - target, profile, runtime, shape, or dtype scope changes
---
# Device win with wall-time loss

## Summary

The retained MSKL workflow reports kernel execution time for tuning configurations without an end-to-end wall-time result in the same example. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Problem or symptom

A locally observed device-time improvement is insufficient to claim an application wall-time improvement unless the same call path is synchronized and measured end to end. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Mechanism

Treat host preparation, compilation, launch, synchronization, generated-artifact handling, and device execution as separate attribution regions. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Applicability

Use this pattern when device-only measurements and user-visible latency can cover different work. The reviewed local counterexample is evidence only for its exact runtime, synchronization policy, host path, shape, dtype, and harness. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md) [Reviewed local Source](../../sources/local/ascend/source-local-ascend-flexattention-round-003.md)

## Implementation approaches

Change measurement and attribution before changing the kernel when the missing interval is outside the device region. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Expected observables

Compare correctness, kernel count, device interval, host interval, synchronization, and synchronized wall time under the same workload. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Risks and counterexamples

Do not label a result a wall loss merely because wall evidence is absent; preserve the status as unknown until measured. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Examples

One reviewed historical, Designer-only counterexample measured exactly Ascend910B4, `triton_ascend`, Triton 3.2.0 / torch_npu 2.7.1.post4, tokens=83, heads=8, head_size=64, kv_heads=8, fp16 inputs/output, and fp32 accumulation. Device time fell from 0.0544268 to 0.0240532 ms/call, while synchronized wall median regressed from 0.296535 to 0.321280 ms (-8.344714789147998%). The result is limited to that runtime, synchronization policy, host path, shape, dtype, Round 003 measurement, and harness. [Reviewed local Source](../../sources/local/ascend/source-local-ascend-flexattention-round-003.md)

## Transfer boundaries

Do not transfer attribution across a different runtime, synchronization policy, host path, cache state, device, or measurement harness. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Required local checks

Measure the current baseline and candidate with explicit synchronization and report device and wall intervals separately. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Sources

- [Reviewed historical attention Round 003](../../sources/local/ascend/source-local-ascend-flexattention-round-003.md)
- [MindStudio Kernel Launcher user guide at f9fbf4d2](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

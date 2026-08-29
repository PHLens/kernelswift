---
schema_version: 1
id: technique-kernel-fusion
title: Kernel fusion as a review hypothesis
type: technique
audiences: [designer]
authority: advisory
summary: Evaluate fusion without promoting the seed corpus into an implementation recipe.
targets: [ascend]
target_match: backend
languages: [ascendc, triton]
kernel_types: []
techniques: [kernel-fusion]
hardware_features: [execution-pipeline, memory-hierarchy]
tags: [kernel-fusion, launch-collapse, materialization]
symptoms: [launch-bound, materialization-overhead]
sources: [source-local-ascend-groupedtopk-round-001, source-mskl-user-guide-f9fbf4d2, source-triton-ascend-readme-865691e2]
related: []
prerequisites: []
version_sensitive: []
observations:
  - id: observation-fusion-evidence-not-established
    text: The retained seed Sources describe compiler pipelines and generated kernel launch code but do not report a reviewed fusion result, so fusion remains a locally tested hypothesis.
    source_id: source-mskl-user-guide-f9fbf4d2
    locator: artifact user guide lines 3-8 and 151-173
    evidence_level: inferred
    reproduction: concept
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [absence of a fusion result cannot establish legality or benefit]
examples:
  - id: example-groupedtopk-fusion-round-001
    role: positive
    subtype: performance
    source_id: source-local-ascend-groupedtopk-round-001
    locator: reviewed proposal observations and transfer boundaries
    evidence_level: source-reported
    reproduction: runnable
    target_id: ascend910b4
    implementation_profile_id: triton_ascend
    profile_authority: historical-noncanonical
    runtime_fingerprint: triton-3.2.0 torch-npu-2.7.1.post4
    operator_family: groupedtopk
    shape:
      E: 256
      GROUPS: 8
      HIDDEN: 7168
      T: 83
      TOPGROUPS: 4
      TOPK: 8
    dtype: fp16
    terminal_classification: accepted
    comparability: historical-local
    measurement_fingerprint: d2dc2d5a61930039371da06149b3156c4911a136c6c5df859f50d68ea0e3b871
    baseline_id: null
    candidate_id: null
    observed:
      - {metric: correctness_pass, value: true, statistic: exact, unit: boolean}
      - {metric: device_time_ms, value: 0.034634, statistic: exact, unit: milliseconds}
      - {metric: kernel_count_per_call, value: 1.0, statistic: exact, unit: count}
      - {metric: wall_improvement_pct, value: 54.88475414304643, statistic: exact, unit: percent}
      - {metric: wall_time_ms, value: 0.32162, statistic: median, unit: milliseconds}
    transfer_boundary: exact Ascend910B4, triton_ascend, Triton 3.2.0 / torch_npu 2.7.1.post4, T=83, E=256, hidden=7168, topk=8, groups=8, topgroups=4, fp16/fp32/int32 semantics, Round 001 measurement, and original harness only
    reconsider_when: [binding:missing-vnext-artifact, sketch:missing-vnext-artifact, verdict:missing-vnext-artifact]
---
# Kernel fusion as a review hypothesis

## Summary

The original seed Sources support examining generated launch code and compiler-managed execution without prescribing fusion. A separate reviewed historical local Source now provides one tightly scoped positive example for Designer use only. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md) [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md) [Reviewed local Source](../../sources/local/ascend/source-local-ascend-groupedtopk-round-001.md)

## Problem or symptom

Use this page when separate work appears launch- or materialization-sensitive; the captured Sources do not establish that fusion is the cause or remedy. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Mechanism

Treat fusion only as a candidate change to the generated or compiled kernel boundary, not as a source-backed implementation prescription. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Applicability

Require an independently legal producer-consumer boundary and a matching target/runtime before experimentation. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Implementation approaches

Preserve algorithm, precision, dataflow, effects, aliases, host plan, and public interface while testing only an implementation-level boundary change. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Expected observables

Measure launch count, device time, and synchronized wall time locally; the seed Sources do not provide a fusion measurement. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Risks and counterexamples

Reject the hypothesis when legality, resources, compilation, device attribution, or wall behavior regresses. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Examples

One reviewed historical, Designer-only local example fused a grouped selection path into one Triton-Ascend kernel for exactly Ascend910B4, `triton_ascend`, Triton 3.2.0 / torch_npu 2.7.1.post4, T=83, E=256, hidden=7168, topk=8, groups=8, topgroups=4, and its original fp16/fp32/int32 semantics. In that exact Round 001 measurement, correctness passed, kernel count became 1 per call, device time was 0.034634 ms/call, and synchronized wall median was 0.321620 ms with a 54.88475414304643% reported improvement. This is historical-noncanonical evidence, not a transferable recipe. [Reviewed local Source](../../sources/local/ascend/source-local-ascend-groupedtopk-round-001.md)

## Transfer boundaries

Do not transfer a fusion conclusion across a different graph, dtype, shape, compiler, runtime, or target. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Required local checks

Run correctness, lowering, resource, launch-count, device-time, and wall-time checks before publication. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Sources

- [Reviewed historical grouped selection Round 001](../../sources/local/ascend/source-local-ascend-groupedtopk-round-001.md)
- [MindStudio Kernel Launcher user guide at f9fbf4d2](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)
- [Triton Ascend README at 865691e2](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

---
schema_version: 1
id: pattern-ascend-capability-gap
title: Ascend capability-gap handling
type: pattern
audiences: [designer]
authority: advisory
summary: Preserve unknown or unsupported backend capabilities instead of inventing recipes.
targets: [ascend]
target_match: backend
languages: [ascendc, triton]
kernel_types: [selection, topk]
techniques: [tiling, work-partitioning]
hardware_features: [memory-hierarchy]
tags: [ascend, capability-gap, device-attribution]
symptoms: [capability-gap]
sources: [source-ascendc-programming-model-cann-900beta1, source-triton-ascend-readme-865691e2]
related: []
prerequisites: []
version_sensitive: []
candidate_techniques: [technique-tiling-and-work-partitioning, technique-topk-selection-and-reduction]
observations:
  - id: observation-ascendc-metadata-only-gap
    text: The official Ascend C Source is metadata-only, so it cannot support a retained implementation recipe or exact capability claim.
    source_id: source-ascendc-programming-model-cann-900beta1
    locator: Source license state and capture facts
    evidence_level: inferred
    reproduction: concept
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [metadata identity is not implementation authority]
  - id: observation-triton-ascend-completeness-gap
    text: The captured Triton-Ascend README says API completeness dtype support memory-access flexibility and automatic optimization are still being improved.
    source_id: source-triton-ascend-readme-865691e2
    locator: artifact README.md lines 13-15
    evidence_level: source-reported
    reproduction: concept
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [a general improvement statement does not classify a specific capability as supported or unsupported]
examples: []
---
# Ascend capability-gap handling

## Summary

The captured Triton-Ascend README explicitly describes areas still being improved, while the official Ascend C Source retains only document identity metadata. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md) [Official metadata](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)

## Problem or symptom

Use this pattern when an exact API, dtype, memory access, lowering, or programming-model authority is missing from the reviewed corpus. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Mechanism

Classify the missing capability as unknown until an allowlisted authority or local probe establishes support or rejection. [Official metadata](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)

## Applicability

Apply the gap only to the exact target, runtime, profile, dtype, shape, and operation under review. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Implementation approaches

Prefer a supported spelling only after authority or probing; otherwise return an empty/capability-gap result instead of an invented Ascend C or Triton recipe. [Official metadata](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)

## Expected observables

Record the capability ID, profile/runtime authority, probe result, lowering evidence, and stable unknown or unsupported status. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Risks and counterexamples

A different backend revision or profile may support the operation, so a gap must remain scope-bound and revisable. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Examples

No fake canonical Ascend profile or Ascend C implementation recipe is included in this seed corpus. [Official metadata](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)

## Transfer boundaries

Do not generalize an unknown or unsupported result across runtime, profile, dtype, shape, operation, or target boundaries. [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Required local checks

Resolve the exact authority or run a bounded capability probe before promoting guidance. [Official metadata](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)

## Sources

- [Official CANN 9.0.0-beta.1 page What is Ascend C](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)
- [Triton Ascend README at 865691e2](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

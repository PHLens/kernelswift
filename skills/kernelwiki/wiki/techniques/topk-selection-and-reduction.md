---
schema_version: 1
id: technique-topk-selection-and-reduction
title: Top-k selection and reduction evidence boundary
type: technique
audiences: [designer]
authority: advisory
summary: Evaluate top-k selection through generic sort reduction and partitioning questions.
targets: [ascend]
target_match: backend
languages: [triton]
kernel_types: [reduction, selection, topk]
techniques: [tiling, work-partitioning]
hardware_features: [memory-hierarchy]
tags: [reduction, selection, topk]
symptoms: [capability-gap]
sources: [source-triton-ascend-readme-865691e2]
related: []
prerequisites: []
version_sensitive: []
observations:
  - id: observation-triton-scan-sort-scope
    text: The captured Triton-Ascend README reports improved Scan and Sort Python APIs and non-contiguous memory access but does not establish an exact top-k implementation contract.
    source_id: source-triton-ascend-readme-865691e2
    locator: artifact README.md lines 35-40
    evidence_level: source-reported
    reproduction: concept
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [Scan and Sort support does not prove a top-k lowering for every dtype shape or runtime]
examples: []
---
# Top-k selection and reduction evidence boundary

## Summary

The captured backend README reports improvements to Scan and Sort APIs, which is relevant context for selection and reduction design but not an exact top-k recipe. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Problem or symptom

Use this page when a design needs bounded selection, ordering, or reduction and the exact backend capability is not yet established. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Mechanism

Decompose the question into candidate partitioning, ordering, reduction, memory-access, and merge stages without assuming a particular lowering. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Applicability

Require the exact dtype, shape regime, ordering semantics, tie behavior, and supported API subset before implementation. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Implementation approaches

Treat tiling and work partitioning as review dimensions; do not infer a Coder-ready top-k spelling from the generic Scan or Sort statement. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Expected observables

Check correctness, selected values and indices, tie handling, lowering, memory traffic, device time, and wall time locally. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Risks and counterexamples

A supported Scan or Sort surface does not establish a top-k implementation for an arbitrary profile or non-contiguous access pattern. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Examples

No operator-specific top-k Card or implementation recipe is included in the seed corpus. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Transfer boundaries

Do not transfer selection behavior across dtype, K, shape, layout, target, runtime, or API-version boundaries without validation. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Required local checks

Seal correctness and ordering expectations first, then verify capability, lowering, device attribution, and synchronized wall behavior. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Sources

- [Triton Ascend README at 865691e2](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

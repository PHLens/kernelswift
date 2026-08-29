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
languages: [cpp, python]
kernel_types: []
techniques: [tiling]
hardware_features: []
tags: [device-win-wall-loss, host-bound, synchronization]
symptoms: [device-win-wall-loss]
sources: [source-mskl-user-guide-f9fbf4d2]
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
examples: []
---
# Device win with wall-time loss

## Summary

The retained MSKL workflow reports kernel execution time for tuning configurations without an end-to-end wall-time result in the same example. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Problem or symptom

A locally observed device-time improvement is insufficient to claim an application wall-time improvement unless the same call path is synchronized and measured end to end. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Mechanism

Treat host preparation, compilation, launch, synchronization, generated-artifact handling, and device execution as separate attribution regions. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Applicability

Use this pattern when device-only measurements and user-visible latency can cover different work; the seed corpus contains no local numeric counterexample. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Implementation approaches

Change measurement and attribution before changing the kernel when the missing interval is outside the device region. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Expected observables

Compare correctness, kernel count, device interval, host interval, synchronization, and synchronized wall time under the same workload. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Risks and counterexamples

Do not label a result a wall loss merely because wall evidence is absent; preserve the status as unknown until measured. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Examples

No reviewed local device-win/wall-loss example is published in Task 8; Phase D may add one only after explicit review. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Transfer boundaries

Do not transfer attribution across a different runtime, synchronization policy, host path, cache state, device, or measurement harness. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Required local checks

Measure the current baseline and candidate with explicit synchronization and report device and wall intervals separately. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Sources

- [MindStudio Kernel Launcher user guide at f9fbf4d2](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

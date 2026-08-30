---
schema_version: 1
id: measurement-cann-device-attribution
title: CANN device-time attribution boundary
type: measurement
audiences: [designer]
authority: advisory
summary: Separate source-reported kernel timing from broader runtime and wall-time conclusions.
targets: [ascend]
target_match: backend
languages: [cpp, python, triton]
kernel_types: []
techniques: [tiling]
hardware_features: []
tags: [cann, device-attribution, profiling]
symptoms: [device-win-wall-loss]
sources: [source-mskl-user-guide-f9fbf4d2, source-triton-ascend-readme-865691e2]
related: []
prerequisites: []
version_sensitive: []
observations:
  - id: observation-mskl-kernel-performance-collection
    text: The reviewed MSKL guide performs warmup and repeated execution on named device IDs and reports kernel execution time for each tuning configuration.
    source_id: source-mskl-user-guide-f9fbf4d2
    locator: artifact user guide lines 308-355
    evidence_level: source-reported
    reproduction: runnable
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [source-reported kernel timings are tied to the documented example and are not end-to-end wall measurements]
  - id: observation-triton-performance-duration-scope
    text: The captured Triton-Ascend README defines its illustrated speedup as an AscendC duration divided by a Triton duration and names the hardware series.
    source_id: source-triton-ascend-readme-865691e2
    locator: artifact README.md lines 47-59
    evidence_level: source-reported
    reproduction: concept
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [the retained README does not include the underlying benchmark records or wall-time attribution]
examples: []
---
# CANN device-time attribution boundary

## Summary

The retained Sources report kernel or operator duration in specific examples, but they do not establish end-to-end wall-time attribution for another project. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md) [Triton source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Measurement scope

The MSKL guide uses warmup, repeat counts, and device IDs while reporting per-configuration kernel execution time. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Comparison scope

The Triton-Ascend README defines an illustrated speedup ratio from two duration values and identifies an Ascend hardware series, but the retained README does not include the raw benchmark artifact. [Source](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

## Attribution boundary

Record correctness, warmup, repeats, device identity, synchronization, device interval, host interval, and the exact measured call path before interpreting a result. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Required local checks

Reproduce the current baseline and candidate under the same harness, then report device and synchronized wall measurements separately. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Sources

- [MindStudio Kernel Launcher user guide at f9fbf4d2](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)
- [Triton Ascend README at 865691e2](../../sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)

---
schema_version: 1
id: runtime-ascend-kernel-integration
title: Ascend kernel integration boundaries
type: runtime
audiences: [designer]
authority: advisory
summary: Navigate source-backed plugin and direct-kernel integration boundaries on Ascend.
targets: [ascend]
target_match: backend
languages: [cpp, python]
kernel_types: []
techniques: []
hardware_features: []
tags: [ascend, integration, launcher]
symptoms: []
sources: [source-mskl-user-guide-f9fbf4d2, source-vllm-ascend-readme-7702ccd7]
related: []
prerequisites: []
version_sensitive: []
observations:
  - id: observation-mskl-direct-kernel-invocation
    text: The reviewed MSKL guide documents direct tiling-function and kernel-binary invocation for prepared operator projects.
    source_id: source-mskl-user-guide-f9fbf4d2
    locator: artifact user guide lines 31-41 and 90-145
    evidence_level: source-reported
    reproduction: runnable
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [requires the documented project artifacts and matching call signature]
  - id: observation-vllm-ascend-plugin-boundary
    text: The captured vLLM Ascend README describes a hardware plugin that decouples Ascend NPU integration from vLLM through a hardware-pluggable interface.
    source_id: source-vllm-ascend-readme-7702ccd7
    locator: artifact README.md lines 50-58
    evidence_level: source-reported
    reproduction: concept
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [does not establish compatibility for an unlisted model runtime or software stack]
examples: []
---
# Ascend kernel integration boundaries

## Summary

The captured vLLM Ascend README describes a hardware plugin boundary for connecting vLLM to Ascend NPUs. [Source](../../sources/commits/vllm-ascend/7702ccd7d8dea6b4dabdacb0118adb522dedbec7.md)

## Direct invocation

The retained MSKL guide documents a separate project-level route for invoking tiling functions and user-defined kernel binaries from Python. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Integration boundary

Neither Source establishes that a kernel can be transplanted between these routes without matching interfaces, software versions, binary artifacts, launch parameters, and supported hardware. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md) [vLLM source](../../sources/commits/vllm-ascend/7702ccd7d8dea6b4dabdacb0118adb522dedbec7.md)

## Required local checks

Verify the host interface, runtime versions, launch signature, target device, input contracts, and observed correctness in the actual integration path. [MSKL source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Sources

- [MindStudio Kernel Launcher user guide at f9fbf4d2](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)
- [vLLM Ascend README at 7702ccd7](../../sources/commits/vllm-ascend/7702ccd7d8dea6b4dabdacb0118adb522dedbec7.md)

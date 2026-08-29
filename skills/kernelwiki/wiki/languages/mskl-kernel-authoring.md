---
schema_version: 1
id: language-mskl-kernel-authoring
title: MSKL kernel authoring and invocation
type: language
audiences: [designer]
authority: advisory
summary: Navigate the reviewed MSKL invocation, generation, compilation, and tuning workflow.
targets: [ascend]
target_match: backend
languages: [cpp, python]
kernel_types: []
techniques: [tiling]
hardware_features: []
tags: [kernel-authoring, mskl]
symptoms: []
sources: [source-mskl-user-guide-f9fbf4d2]
related: []
prerequisites: []
version_sensitive: []
observations:
  - id: observation-mskl-authoring-workflow
    text: The reviewed guide documents direct invocation of msOpGen tiling and kernel functions and generation, compilation, execution, replacement, and autotuning of template-library kernel launch code.
    source_id: source-mskl-user-guide-f9fbf4d2
    locator: artifact user guide lines 3-8 and 31-41
    evidence_level: source-reported
    reproduction: runnable
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [requires the documented project form environment and matching kernel interface]
examples: []
---
# MSKL kernel authoring and invocation

## Summary

The reviewed MSKL guide documents lightweight invocation of msOpGen tiling and user-defined kernel functions. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Authoring flow

The guide presents Python-driven launch-code generation, compilation, kernel execution, code replacement, and autotuning for template-library kernels. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Boundary

The documented examples depend on prepared CANN software, matching host and kernel interfaces, generated artifacts, and supported project layouts; they are not a universal recipe for another project. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Required local checks

Confirm the exact project template, binary, tiling interface, generated launch code, input types, and device environment before reuse. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Sources

- [MindStudio Kernel Launcher user guide at f9fbf4d2](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

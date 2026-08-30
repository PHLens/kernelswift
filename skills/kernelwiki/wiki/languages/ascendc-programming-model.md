---
schema_version: 1
id: language-ascendc-programming-model
title: Ascend C programming-model evidence boundary
type: language
audiences: [designer]
authority: advisory
summary: Navigate the official Ascend C document identity and reviewed host-kernel authoring evidence.
targets: [ascend]
target_match: backend
languages: [ascendc, cpp]
kernel_types: []
techniques: [tiling]
hardware_features: [memory-hierarchy]
tags: [ascend, ascendc, kernel-authoring, memory-hierarchy]
symptoms: [capability-gap]
sources: [source-ascendc-programming-model-cann-900beta1, source-mskl-user-guide-f9fbf4d2]
related: []
prerequisites: []
version_sensitive: []
observations:
  - id: observation-ascendc-official-document-identity
    text: The captured official metadata pins the CANN Community Edition 900beta1 page titled What is Ascend C, but retains no prose because its license was not explicitly approved.
    source_id: source-ascendc-programming-model-cann-900beta1
    locator: Source title URL and capture facts
    evidence_level: source-reported
    reproduction: concept
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [metadata-only evidence cannot establish an implementation recipe]
  - id: observation-mskl-host-kernel-boundary
    text: The reviewed MSKL guide separates Host-side preparation and Kernel-side implementation and invokes tiling and kernel functions through documented interfaces.
    source_id: source-mskl-user-guide-f9fbf4d2
    locator: artifact user guide lines 31-41 and 76-130
    evidence_level: source-reported
    reproduction: runnable
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [does not define the complete Ascend C language or memory model]
examples: []
---
# Ascend C programming-model evidence boundary

## Summary

The official CANN metadata capture pins the 900beta1 document identity for the page titled “What is Ascend C,” but its prose is intentionally not retained under a metadata-only license decision. [Source](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)

## Reviewed authoring evidence

The retained MSKL guide shows a Host-side and Kernel-side project workflow, tiling invocation, kernel-binary invocation, generated launch code, compilation, and execution. [Source](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Evidence boundary

These Sources support navigation and project-bound authoring concepts only; they do not establish a complete language specification, exact memory hierarchy, or Coder-ready implementation recipe. [Official metadata](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md) [Reviewed guide](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

## Required local checks

Consult an approved authoritative manual and validate the exact CANN revision, target, interfaces, tiling data, memory operations, and generated binary before implementation. [Official metadata](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)

## Sources

- [Official CANN 9.0.0-beta.1 page What is Ascend C](../../sources/docs/source-ascendc-programming-model-cann-900beta1.md)
- [MindStudio Kernel Launcher user guide at f9fbf4d2](../../sources/docs/source-mskl-user-guide-f9fbf4d2.md)

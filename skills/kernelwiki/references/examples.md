# KernelWiki v1 complete examples

These records match `tests/fixtures/valid-corpus/` and pass `scripts/validate.py`.

## Source

```markdown
---
schema_version: 1
id: source-valid-manual
source_kind: manual-doc
title: Reviewed fusion note
url: https://example.invalid/kernel-fusion
repository_id: local
captured_at: "2026-08-21T00:00:00Z"
target_disposition: backend
languages: [triton]
kernel_types: [reduction]
techniques: [kernel-fusion]
hardware_features: [memory-hierarchy]
tags: [kernel-fusion, memory-hierarchy]
license_state: metadata-only
audiences: [designer]
---
# Reviewed fusion note

This metadata-only source records a reviewed claim about legal producer-consumer fusion boundaries.
```

## Generic technique Card

```markdown
---
schema_version: 1
id: technique-kernel-fusion
title: Kernel fusion
type: technique
audiences: [designer]
authority: advisory
summary: Fuse legal producer-consumer work only when the boundary remains valid.
targets: [ascend]
target_match: backend
languages: [triton]
kernel_types: [reduction]
techniques: [kernel-fusion]
hardware_features: [memory-hierarchy]
tags: [kernel-fusion, memory-hierarchy]
symptoms: [launch-bound, materialization-overhead]
sources: [source-valid-manual]
related: []
prerequisites: []
version_sensitive: []
observations:
  - id: observation-fusion-launch-count
    text: Fusion can reduce separately materialized producer-consumer launches in the cited implementation.
    source_id: source-valid-manual
    locator: Reviewed fusion note
    evidence_level: source-reported
    reproduction: concept
    targets: [ascend]
    target_match: backend
    implementation_profile_id: null
    runtime_fingerprint: null
    versions: []
    transfer_boundaries: [requires an independently legal fusion boundary]
examples:
  - id: example-source-fusion
    role: positive
    subtype: source-example
    source_id: source-valid-manual
    locator: Reviewed fusion note
    evidence_level: source-reported
    reproduction: concept
    target_id: ascend
    implementation_profile_id: null
    profile_authority: source-only
    runtime_fingerprint: null
    operator_family: reduction
    shape: {elements: N}
    dtype: fp32
    terminal_classification: source-reported
    comparability: source-reported
    measurement_fingerprint: null
    baseline_id: null
    candidate_id: null
    observed:
      - {metric: kernel_count_per_call, value: 1, statistic: source-reported, unit: count}
    transfer_boundary: The source does not establish legality for a different producer-consumer graph.
    reconsider_when: [the fusion boundary or runtime changes]
---
# Kernel fusion

## Summary

Fuse legal producer-consumer work to reduce avoidable launch and materialization overhead.

## Problem or symptom

Separate launches can dominate small producer-consumer chains.

## Mechanism

Place compatible work in one kernel without changing semantics.

## Applicability

Use only when indexing, precision, effects, and aliases remain legal.

## Implementation approaches

Preserve the public interface and choose an implementation-specific spelling.

## Expected observables

Launch count can decrease for the matched call path.

## Risks and counterexamples

Fusion can increase register pressure or move cost outside the measured device region.

## Examples

The reviewed source records a source-reported reduction example.

## Transfer boundaries

Do not generalize across target, runtime, or shape without local qualification.

## Required local checks

Re-run correctness, lowering, device attribution, and synchronized wall measurement.

## Sources

- `source-valid-manual`
```

The complete positive, counterexample, and capability-gap example records used to exercise role conditionals are checked in at `tests/fixtures/valid-corpus/examples.yaml`.

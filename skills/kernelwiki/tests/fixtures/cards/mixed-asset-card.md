---
schema_version: 1
id: language-mixed-asset
title: Exact-profile page with mixed assets
type: language
audiences: [coder, designer]
authority: advisory
summary: An exact MLU guidance page with one approved snippet and one non-eligible full kernel.
targets: [mlu590]
target_match: exact
languages: [triton]
kernel_types: [reduction, topk]
techniques: [tiling]
hardware_features: [memory-hierarchy]
tags: [reduction, selection, tiling, topk]
symptoms: [memory-bound]
sources: [source-exact-coder]
related: []
prerequisites: []
version_sensitive: []
observations: []
examples:
  - id: example-test-exact
    role: positive
    subtype: source-example
    source_id: source-exact-coder
    locator: Exact test fixture
    evidence_level: source-reported
    reproduction: runnable
    target_id: mlu590
    implementation_profile_id: triton_mlu
    profile_authority: current-vnext
    runtime_fingerprint: triton 3.6.0 / CoreX 4.4.0
    operator_family: topk
    shape: {E: 256, K: 8, T: 83}
    dtype: fp32
    terminal_classification: source-reported
    comparability: current-contract
    measurement_fingerprint: null
    baseline_id: null
    candidate_id: null
    observed:
      - {metric: correctness_pass, value: true, statistic: exact, unit: boolean}
    transfer_boundary: Exact MLU profile, runtime, dtype, and shape only.
    reconsider_when: [profile or runtime changes]
---
# Exact-profile page with mixed assets

The page is readable only after exact role admission. Asset access remains independent.

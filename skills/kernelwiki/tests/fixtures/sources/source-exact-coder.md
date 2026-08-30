---
schema_version: 1
id: source-exact-coder
source_kind: github-commit
title: Exact MLU Coder fixture source
url: https://github.com/Ascend/triton-ascend/commit/1111111111111111111111111111111111111111
repository_id: triton-ascend
captured_at: '2026-08-21T00:00:00Z'
target_disposition: exact
target_ids: [mlu590]
languages: [triton]
kernel_types: [reduction, topk]
techniques: [tiling]
hardware_features: [memory-hierarchy]
tags: [reduction, selection, tiling, topk]
license_state: approved
artifact_dir: artifacts/source-exact-coder
implementation_profile_ids: [triton_mlu]
runtime_fingerprints: [triton 3.6.0 / CoreX 4.4.0]
audiences: [coder, designer]
---
# Exact MLU Coder fixture source

This checked-in test Source is scoped to the exact validated MLU profile and runtime fixture.

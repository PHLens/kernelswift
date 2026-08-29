---
schema_version: 1
id: source-local-s60-mm-encoder-attention-round-002
source_kind: local-campaign
title: 'Reviewed historical local campaign evidence: kernels/track1-triton/mm_encoder_attention/s60/epoch2#round-002'
url: local://4ff2094d96c66fde22192f8283113228123f7397/kernels/track1-triton/mm_encoder_attention/s60/epoch2
repository_id: local
captured_at: '2026-08-29T17:00:00Z'
target_disposition: exact
target_ids:
- s60
implementation_profile_ids:
- triton_gcu
runtime_fingerprints:
- triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2
languages:
- python
- triton
kernel_types:
- attention
techniques:
- tiling
hardware_features:
- vector
tags:
- device-win-wall-loss
- gcu
- s60
- tiling
- triton-gcu
license_state: unknown
audiences:
- designer
profile_authority: historical-noncanonical
strict_vnext_validated: false
missing_evidence:
- verdict-round-profile-target-not-modeled
---
# Reviewed historical local campaign evidence: kernels/track1-triton/mm_encoder_attention/s60/epoch2#round-002

Immutable reviewed historical campaign evidence. This Source is metadata-only, Designer-only, and not validated against the current vNext contract.

## Curator review

```json
{
  "decision": "include",
  "proposal_id": "experience-historical-source-local-s60-mm-encoder-attention-round-002",
  "proposal_sha256": "a28008509845a877587b6108326871142e12e7892cdc7e6405f61c4c4a5db5e1",
  "publication_target": {
    "card_id": "pattern-device-win-wall-loss",
    "mode": "existing-card-example"
  },
  "rationale": "Include as Designer-only counterexample example of pattern-device-win-wall-loss. S60 (Enflame GCU), triton_gcu, round-002.",
  "reviewed_at": "2026-08-29T17:00:00Z",
  "reviewed_by": "kernelwiki-curator"
}
```

## Reviewed proposal evidence

```json
{
  "artifact_hashes": {
    "base": "86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2",
    "candidate": "7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818",
    "decision": "04f6dc0b6a92429ba7538d2dfa3d6c4e10471a05d80188a716d5770e2f031e2f",
    "report": "78ceeaa38407b5a01a40b8d661218b2cf31265a10ec3931b139853d55c50c25f",
    "sketch": "c3c585d1f95337f25ac1c9ff5dc3c3591637b1e9a7c906174fb60d0da97695dd",
    "verdict": "f4cf8c2b9ed249fcfc3f00a00daf5933c3d385fbc42aa00c68da98b8dc3f0105"
  },
  "missing_evidence": [
    "verdict-round-profile-target-not-modeled"
  ],
  "observed": [
    {
      "evidence_ref": "kernels/track1-triton/mm_encoder_attention/s60/epoch2/rounds/report_002.md",
      "metric": "correctness_pass",
      "statistic": "exact",
      "unit": "boolean",
      "value": true
    },
    {
      "evidence_ref": "kernels/track1-triton/mm_encoder_attention/s60/epoch2/rounds/report_002.md",
      "metric": "wall_improvement_pct",
      "statistic": "exact",
      "unit": "percent",
      "value": -9.3151
    },
    {
      "evidence_ref": "kernels/track1-triton/mm_encoder_attention/s60/epoch2/rounds/report_002.md",
      "metric": "wall_time_ms",
      "statistic": "median",
      "unit": "milliseconds",
      "value": 0.275038
    }
  ],
  "scope": {
    "allowed_audiences": [
      "designer"
    ],
    "asset_mode": "metadata-only",
    "audiences": [
      "designer"
    ],
    "captured_at": "2026-08-29T17:00:00Z",
    "comparability": "historical-local",
    "hardware_features": [
      "vector"
    ],
    "implementation_profile_id": "triton_gcu",
    "kernel_types": [
      "attention"
    ],
    "languages": [
      "python",
      "triton"
    ],
    "license_state": "unknown",
    "measurement_fingerprint": null,
    "profile_authority": "historical-noncanonical",
    "repository_id": "local",
    "source_id": "source-local-s60-mm-encoder-attention-round-002",
    "tags": [
      "device-win-wall-loss",
      "gcu",
      "s60",
      "tiling",
      "triton-gcu"
    ],
    "target_id": "s60",
    "techniques": [
      "tiling"
    ]
  },
  "terminal": {
    "commit": "4ff2094d96c66fde22192f8283113228123f7397",
    "local_locator": "kernels/track1-triton/mm_encoder_attention/s60/epoch2#round-002",
    "project_path": "kernels/track1-triton/mm_encoder_attention/s60/epoch2",
    "result": "no-improvement",
    "strict_vnext_validated": false
  },
  "transfer_boundaries": [
    "target=s60",
    "profile=triton_gcu historical-noncanonical",
    "runtime=triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2",
    "shape=BATCH2xHEAD8xSEQ83xHEAD_SIZE64",
    "dtype=fp16",
    "round=002"
  ]
}
```

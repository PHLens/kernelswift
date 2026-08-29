---
schema_version: 1
id: source-local-s60-flexattention-round-001
source_kind: local-campaign
title: 'Reviewed historical local campaign evidence: kernels/track1-triton/flexattention/s60/epoch2#round-001'
url: local://4ff2094d96c66fde22192f8283113228123f7397/kernels/track1-triton/flexattention/s60/epoch2
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
# Reviewed historical local campaign evidence: kernels/track1-triton/flexattention/s60/epoch2#round-001

Immutable reviewed historical campaign evidence. This Source is metadata-only, Designer-only, and not validated against the current vNext contract.

## Curator review

```json
{
  "decision": "include",
  "proposal_id": "experience-historical-source-local-s60-flexattention-round-001",
  "proposal_sha256": "9af07b6c117442f8aaae805be7c447b912fb86495660ea0f5b2268323eda1e7f",
  "publication_target": {
    "card_id": "pattern-device-win-wall-loss",
    "mode": "existing-card-example"
  },
  "rationale": "Include as Designer-only counterexample example of pattern-device-win-wall-loss. S60 (Enflame GCU), triton_gcu, round-001.",
  "reviewed_at": "2026-08-29T17:00:00Z",
  "reviewed_by": "kernelwiki-curator"
}
```

## Reviewed proposal evidence

```json
{
  "artifact_hashes": {
    "base": "dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0",
    "candidate": "6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9",
    "decision": "8a2bb5a7a6bcd2ccb8ecb704c30c5edbb540fb5c52fc4cae34f2afeef57c5d86",
    "report": "014f37e37a462b6239dee542698c60595cd7f36340ac13ff7d0d186c98a64acd",
    "sketch": "aad322a8b806d9f97bc9c5056c8ae1ea62c5bd8ecc8bb502fb6fc72399a61247",
    "verdict": "e48991da22ccb1d441b53fcb1e85bfc2fb2d982f6bc846bd334e53bbc76e9b64"
  },
  "missing_evidence": [
    "verdict-round-profile-target-not-modeled"
  ],
  "observed": [
    {
      "evidence_ref": "kernels/track1-triton/flexattention/s60/epoch2/rounds/report_001.md",
      "metric": "correctness_pass",
      "statistic": "exact",
      "unit": "boolean",
      "value": true
    },
    {
      "evidence_ref": "kernels/track1-triton/flexattention/s60/epoch2/rounds/report_001.md",
      "metric": "wall_improvement_pct",
      "statistic": "exact",
      "unit": "percent",
      "value": -6.3954
    },
    {
      "evidence_ref": "kernels/track1-triton/flexattention/s60/epoch2/rounds/report_001.md",
      "metric": "wall_time_ms",
      "statistic": "median",
      "unit": "milliseconds",
      "value": 0.266835
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
    "source_id": "source-local-s60-flexattention-round-001",
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
    "local_locator": "kernels/track1-triton/flexattention/s60/epoch2#round-001",
    "project_path": "kernels/track1-triton/flexattention/s60/epoch2",
    "result": "no-improvement",
    "strict_vnext_validated": false
  },
  "transfer_boundaries": [
    "target=s60",
    "profile=triton_gcu historical-noncanonical",
    "runtime=triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2",
    "shape=HEAD8xHEAD_SIZE64xTOKENS83",
    "dtype=fp16",
    "round=001"
  ]
}
```
